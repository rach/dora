//! Embedder layer: trait + two impls (local fastembed, remote OpenAI) + config-driven factory.
//!
//! The trait is sync. dora is a single-operation-at-a-time CLI; making the embedder async would
//! force awkward bridges in the rest of the (sync) call graph. OpenAI's HTTP calls use
//! `reqwest::blocking`, which internally manages its own runtime — no tokio surfaces to callers.

use anyhow::{anyhow, bail, Context, Result};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread::sleep;
use std::time::Duration;

use crate::config::EmbedderConfig;

pub trait Embedder: Send + Sync {
    fn id(&self) -> &str;
    fn dims(&self) -> usize;
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let v = vec![text.to_string()];
        let mut out = self.embed(&v)?;
        out.pop().ok_or_else(|| anyhow!("empty embed result"))
    }
    /// USD per million input tokens. `None` = free / local. Used by main for the cost preview.
    fn cost_per_million_tokens(&self) -> Option<f64> {
        None
    }
}

/// Shared by reference across sources that pick the same model — see `mcp::run_multi`'s
/// embedder cache. Single-source callers just hold one strong ref; cost is negligible.
pub type DynEmbedder = Arc<dyn Embedder>;

pub fn from_config(cfg: &EmbedderConfig, models_dir: &Path) -> Result<DynEmbedder> {
    match cfg.provider.as_str() {
        "fastembed" => {
            let inner = FastembedEmbedder::new(&cfg.model, models_dir.to_path_buf())?;
            Ok(Arc::new(inner))
        }
        "openai" => {
            let inner = OpenAIEmbedder::new(&cfg.model, &cfg.api_key_env, cfg.dimensions)?;
            Ok(Arc::new(inner))
        }
        other => bail!(
            "unknown embedder provider '{other}'. supported: fastembed, openai"
        ),
    }
}

/// Canonical cache key for embedder sharing across sources. Two sources with the same
/// (provider, model, dimensions) tuple get back the same `Arc<dyn Embedder>`.
pub fn cache_key(cfg: &EmbedderConfig) -> String {
    format!(
        "{}|{}|{}",
        cfg.provider,
        cfg.model,
        cfg.dimensions.map(|d| d.to_string()).unwrap_or_default(),
    )
}

// ---------------- fastembed (local ONNX) ----------------

pub struct FastembedEmbedder {
    inner: TextEmbedding,
    id: String,
    dims: usize,
}

impl FastembedEmbedder {
    pub fn new(model_name: &str, model_cache_dir: PathBuf) -> Result<Self> {
        let (model, dims, canonical_code) = resolve_fastembed_model(model_name)?;
        std::fs::create_dir_all(&model_cache_dir).ok();
        let inner = TextEmbedding::try_new(
            InitOptions::new(model)
                .with_cache_dir(model_cache_dir)
                .with_show_download_progress(true),
        )?;
        Ok(Self {
            inner,
            id: format!("fastembed:{canonical_code}"),
            dims,
        })
    }
}

impl Embedder for FastembedEmbedder {
    fn id(&self) -> &str {
        &self.id
    }
    fn dims(&self) -> usize {
        self.dims
    }
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(self.inner.embed(texts.to_vec(), None)?)
    }
}

fn resolve_fastembed_model(name: &str) -> Result<(EmbeddingModel, usize, String)> {
    let want = name.to_lowercase();
    let models = TextEmbedding::list_supported_models();

    let mut full_match: Option<&fastembed::ModelInfo<EmbeddingModel>> = None;
    let mut short_match: Option<&fastembed::ModelInfo<EmbeddingModel>> = None;
    for info in &models {
        let short = short_name(&info.model_code);
        let full = info.model_code.to_lowercase();
        if want == full {
            if full_match.is_none() || prefer_over(info, full_match.unwrap()) {
                full_match = Some(info);
            }
        } else if want == short {
            if short_match.is_none() || prefer_over(info, short_match.unwrap()) {
                short_match = Some(info);
            }
        }
    }
    if let Some(info) = full_match.or(short_match) {
        return Ok((info.model.clone(), info.dim, info.model_code.clone()));
    }

    let mut available: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for info in &models {
        available.insert(short_name(&info.model_code));
    }
    let available: Vec<String> = available.into_iter().collect();
    Err(anyhow!(
        "unknown fastembed model '{name}'. available: {}",
        available.join(", ")
    ))
}

fn short_name(model_code: &str) -> String {
    model_code.rsplit('/').next().unwrap_or("").to_lowercase()
}

fn prefer_over(
    candidate: &fastembed::ModelInfo<EmbeddingModel>,
    current: &fastembed::ModelInfo<EmbeddingModel>,
) -> bool {
    let cand_q = candidate.model_file.contains("quantized");
    let curr_q = current.model_file.contains("quantized");
    !cand_q && curr_q
}

// ---------------- openai (remote HTTP) ----------------

pub struct OpenAIEmbedder {
    client: reqwest::blocking::Client,
    api_key: String,
    model: String,
    id: String,
    dims: usize,
    dimensions_param: Option<usize>,
    cost_per_million: f64,
}

const OPENAI_ENDPOINT: &str = "https://api.openai.com/v1/embeddings";
const OPENAI_BATCH_SIZE: usize = 64;

#[derive(Clone)]
struct OpenAIModelSpec {
    default_dim: usize,
    custom_dim_range: Option<(usize, usize)>,
    cost_per_million: f64,
}

fn openai_model_table() -> Vec<(&'static str, OpenAIModelSpec)> {
    vec![
        (
            "text-embedding-3-small",
            OpenAIModelSpec {
                default_dim: 1536,
                custom_dim_range: Some((256, 1536)),
                cost_per_million: 0.02,
            },
        ),
        (
            "text-embedding-3-large",
            OpenAIModelSpec {
                default_dim: 3072,
                custom_dim_range: Some((256, 3072)),
                cost_per_million: 0.13,
            },
        ),
        (
            "text-embedding-ada-002",
            OpenAIModelSpec {
                default_dim: 1536,
                custom_dim_range: None,
                cost_per_million: 0.10,
            },
        ),
    ]
}

impl OpenAIEmbedder {
    pub fn new(model: &str, api_key_env: &str, dimensions: Option<usize>) -> Result<Self> {
        let table = openai_model_table();
        let spec = table
            .iter()
            .find(|(n, _)| *n == model)
            .map(|(_, s)| s.clone())
            .ok_or_else(|| {
                let supported: Vec<&str> = table.iter().map(|(n, _)| *n).collect();
                anyhow!(
                    "unknown openai model '{model}'. supported: {}",
                    supported.join(", ")
                )
            })?;

        let dims = match (dimensions, spec.custom_dim_range) {
            (None, _) => spec.default_dim,
            (Some(d), Some((mn, mx))) => {
                if d < mn || d > mx {
                    bail!(
                        "openai model '{model}' supports dimensions {mn}..={mx}, got {d}"
                    );
                }
                d
            }
            (Some(_), None) => bail!(
                "openai model '{model}' does not support custom dimensions"
            ),
        };

        let api_key = std::env::var(api_key_env).with_context(|| {
            format!(
                "openai embedder requires env var {api_key_env} to be set \
                 (e.g. `export {api_key_env}=sk-...`)"
            )
        })?;

        let mut id = format!("openai:{model}");
        if dimensions.is_some() {
            id.push_str(&format!(":dims={dims}"));
        }

        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .context("building reqwest client")?;

        Ok(Self {
            client,
            api_key,
            model: model.to_string(),
            id,
            dims,
            dimensions_param: dimensions,
            cost_per_million: spec.cost_per_million,
        })
    }

    fn embed_batch(&self, batch: &[String]) -> Result<Vec<Vec<f32>>> {
        #[derive(Serialize)]
        struct Req<'a> {
            model: &'a str,
            input: &'a [String],
            #[serde(skip_serializing_if = "Option::is_none")]
            dimensions: Option<usize>,
        }
        #[derive(Deserialize)]
        struct RespItem {
            embedding: Vec<f32>,
            index: usize,
        }
        #[derive(Deserialize)]
        struct Resp {
            data: Vec<RespItem>,
        }

        let body = Req {
            model: &self.model,
            input: batch,
            dimensions: self.dimensions_param,
        };

        let max_attempts = 4;
        let mut delay_ms: u64 = 200;
        let mut last_err: Option<anyhow::Error> = None;

        for attempt in 1..=max_attempts {
            let send_result = self
                .client
                .post(OPENAI_ENDPOINT)
                .bearer_auth(&self.api_key)
                .json(&body)
                .send();

            let resp = match send_result {
                Ok(r) => r,
                Err(e) => {
                    last_err = Some(anyhow::Error::new(e).context("openai request send failed"));
                    if attempt < max_attempts {
                        sleep(Duration::from_millis(delay_ms));
                        delay_ms = (delay_ms * 2).min(4000);
                        continue;
                    }
                    break;
                }
            };

            let status = resp.status();
            if status.is_success() {
                let parsed: Resp = resp.json().context("parsing openai response")?;
                if parsed.data.len() != batch.len() {
                    bail!(
                        "openai returned {} vectors for {} inputs",
                        parsed.data.len(),
                        batch.len()
                    );
                }
                let mut sorted = parsed.data;
                sorted.sort_by_key(|i| i.index);
                let mut out = Vec::with_capacity(sorted.len());
                for item in sorted {
                    if item.embedding.len() != self.dims {
                        bail!(
                            "openai returned vector of length {} (expected {})",
                            item.embedding.len(),
                            self.dims
                        );
                    }
                    out.push(item.embedding);
                }
                return Ok(out);
            }

            let retryable = status.as_u16() == 429 || status.is_server_error();
            if retryable && attempt < max_attempts {
                let wait_ms = resp
                    .headers()
                    .get("retry-after")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|s| s.parse::<u64>().ok())
                    .map(|s| s * 1000)
                    .unwrap_or(delay_ms);
                sleep(Duration::from_millis(wait_ms));
                delay_ms = (delay_ms * 2).min(4000);
                continue;
            }

            let body_excerpt = resp
                .text()
                .unwrap_or_default()
                .chars()
                .take(300)
                .collect::<String>();
            bail!("openai returned status {}: {}", status, body_excerpt);
        }

        Err(last_err.unwrap_or_else(|| {
            anyhow!("openai embedding failed after {max_attempts} attempts")
        }))
    }
}

impl Embedder for OpenAIEmbedder {
    fn id(&self) -> &str {
        &self.id
    }
    fn dims(&self) -> usize {
        self.dims
    }
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(OPENAI_BATCH_SIZE) {
            let mut batch_out = self.embed_batch(chunk)?;
            out.append(&mut batch_out);
        }
        Ok(out)
    }
    fn cost_per_million_tokens(&self) -> Option<f64> {
        Some(self.cost_per_million)
    }
}

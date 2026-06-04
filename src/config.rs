//! User-facing TOML config at `<source>/.dora/config.toml`.
//!
//! Two-layer resolution:
//!   1. Parse the file (any field optional) into [`RawConfig`].
//!   2. Resolve `[source] mode` (auto-detect if unset) → apply mode defaults → overlay explicit
//!      user values from the file → final [`Config`].
//!
//! Most users only set `[source] mode = "..."` (or nothing at all, and let auto-detect fire).
//! Power users override individual sections as needed.

use anyhow::Result;
use serde::Deserialize;
use std::path::Path;

use crate::mode::Mode;

pub const SCHEMA_VERSION: &str = "3";
pub const CHUNKER_VERSION: &str = "4";

// ---------- final, resolved types (what the rest of the codebase reads) ----------

#[derive(Debug, Clone)]
pub struct Config {
    pub source: SourceConfig,
    pub vault: VaultConfig,
    pub chunking: ChunkingConfig,
    pub search: SearchConfig,
    pub embedder: EmbedderConfig,
    pub claude_code: ClaudeCodeConfig,
    pub codex: CodexConfig,
    pub graph: GraphConfig,
    pub confidence: ConfidenceConfig,
}

/// `[confidence]` settings. `ann_floor` is the top-ANN-cosine threshold below which a result
/// set is flagged low-confidence (absent literal/file-agreement evidence). When unset, the
/// effective floor falls back to a per-embedder/mode default — `dora calibrate` derives a real
/// value from the source's own index and writes it here.
#[derive(Debug, Clone, Default)]
pub struct ConfidenceConfig {
    pub ann_floor: Option<f32>,
}

impl Config {
    /// The ANN-cosine floor to use for the low-confidence gate: the calibrated/configured value
    /// if present, otherwise a provisional per-embedder-family / per-mode default.
    pub fn effective_ann_floor(&self) -> f32 {
        self.confidence
            .ann_floor
            .unwrap_or_else(|| default_ann_floor(&self.embedder.model, &self.source.mode))
    }
}

/// Provisional per-embedder-family / per-mode cosine floors, used until `dora calibrate`
/// writes a data-derived value. Deliberately conservative; these are guesses, not gospel.
fn default_ann_floor(model: &str, mode: &str) -> f32 {
    let m = model.to_lowercase();
    if mode == "code" || (m.contains("jina") && m.contains("code")) {
        0.40
    } else if m.contains("embeddinggemma") || m.contains("gemma") {
        0.50
    } else {
        // bge-*, e5-*, minilm, openai, and anything else → the shared default.
        crate::search::DEFAULT_ANN_FLOOR
    }
}

/// `[graph]` settings (Layer B derived edges). `entities` opts into the GLiNER entity-edge
/// extractor (~300 MB ONNX); off by default — keyphrase + similarity edges are always built.
#[derive(Debug, Clone, Default)]
pub struct GraphConfig {
    pub entities: bool,
}

#[derive(Debug, Clone)]
pub struct SourceConfig {
    /// Resolved mode as a canonical string (`"obsidian"` / `"notes"` / `"docs"` / `"code"`).
    /// Never `"auto"` after resolution — that's only valid as a user input.
    pub mode: String,
}

#[derive(Debug, Clone)]
pub struct VaultConfig {
    pub ignore: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ChunkingConfig {
    pub target_bytes: usize,
    pub atomic_below_bytes: usize,
    pub overlap_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct SearchConfig {
    pub top_k: usize,
    pub collapse_per_file: bool,
}

/// Settings specific to `mode = "claude-code"`. Used by `src/chunk/claude_code.rs`
/// and by the active-session settle filter in `run_incremental_index`.
#[derive(Debug, Clone)]
pub struct ClaudeCodeConfig {
    /// Include `thinking` blocks in the synthesized chunk text. Default false — they're often
    /// huge internal-reasoning passages that bloat embeddings without improving retrieval.
    pub include_thinking: bool,
    /// Skip JSONL files whose mtime is newer than this many seconds. The active session is
    /// being written constantly; re-embedding on every flush burns the embedder. Default 60s.
    pub settle_seconds: u64,
}

impl Default for ClaudeCodeConfig {
    fn default() -> Self {
        Self {
            include_thinking: false,
            settle_seconds: 60,
        }
    }
}

/// Settings specific to `mode = "codex"`. Mirror of `ClaudeCodeConfig` — the two share the
/// settle-window pattern but the analog of `thinking` blocks is called `reasoning` in Codex
/// transcripts, hence the separate field name.
#[derive(Debug, Clone)]
pub struct CodexConfig {
    /// Include `reasoning` blocks in the synthesized chunk text. Default false — same
    /// rationale as `[claude_code] include_thinking`.
    pub include_reasoning: bool,
    /// Skip session JSONL files whose mtime is newer than this many seconds. Default 60s.
    pub settle_seconds: u64,
}

impl Default for CodexConfig {
    fn default() -> Self {
        Self {
            include_reasoning: false,
            settle_seconds: 60,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EmbedderConfig {
    /// `"fastembed"` (default) or `"openai"`.
    pub provider: String,
    /// Model name. For fastembed: short name or full HF path. For openai: one of
    /// `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002`.
    pub model: String,
    /// Name of the env var holding the provider API key. Default `"OPENAI_API_KEY"`.
    pub api_key_env: String,
    /// Optional vector dimension override (openai's text-embedding-3-* only). When set,
    /// participates in the canonical embedder id so changing it triggers a clean reindex.
    pub dimensions: Option<usize>,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            top_k: 10,
            collapse_per_file: true,
        }
    }
}

// ---------- raw, partial types parsed from the TOML file ----------

#[derive(Debug, Deserialize, Default, Clone)]
#[serde(default, deny_unknown_fields)]
struct RawConfig {
    source: RawSourceConfig,
    vault: RawVaultConfig,
    chunking: RawChunkingConfig,
    search: RawSearchConfig,
    embedder: RawEmbedderConfig,
    claude_code: RawClaudeCodeConfig,
    codex: RawCodexConfig,
    graph: RawGraphConfig,
    confidence: RawConfidenceConfig,
}

#[derive(Debug, Deserialize, Default, Clone)]
#[serde(default, deny_unknown_fields)]
struct RawConfidenceConfig {
    ann_floor: Option<f32>,
}

#[derive(Debug, Deserialize, Default, Clone)]
#[serde(default, deny_unknown_fields)]
struct RawGraphConfig {
    entities: Option<bool>,
}

#[derive(Debug, Deserialize, Default, Clone)]
#[serde(default, deny_unknown_fields)]
struct RawSourceConfig {
    mode: Option<String>,
}

#[derive(Debug, Deserialize, Default, Clone)]
#[serde(default, deny_unknown_fields)]
struct RawVaultConfig {
    ignore: Option<Vec<String>>,
    /// Additional file extensions to walk, beyond the mode's defaults. Lets a code-mode user
    /// also include `.toml` / `.md` etc. without disabling the rest.
    extensions: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, Default, Clone)]
#[serde(default, deny_unknown_fields)]
struct RawChunkingConfig {
    target_bytes: Option<usize>,
    atomic_below_bytes: Option<usize>,
    overlap_bytes: Option<usize>,
}

#[derive(Debug, Deserialize, Default, Clone)]
#[serde(default, deny_unknown_fields)]
struct RawSearchConfig {
    top_k: Option<usize>,
    collapse_per_file: Option<bool>,
}

#[derive(Debug, Deserialize, Default, Clone)]
#[serde(default, deny_unknown_fields)]
struct RawEmbedderConfig {
    provider: Option<String>,
    model: Option<String>,
    api_key_env: Option<String>,
    dimensions: Option<usize>,
}

#[derive(Debug, Deserialize, Default, Clone)]
#[serde(default, deny_unknown_fields)]
struct RawClaudeCodeConfig {
    include_thinking: Option<bool>,
    settle_seconds: Option<u64>,
}

#[derive(Debug, Deserialize, Default, Clone)]
#[serde(default, deny_unknown_fields)]
struct RawCodexConfig {
    include_reasoning: Option<bool>,
    settle_seconds: Option<u64>,
}

// ---------- loader ----------

impl Config {
    /// Load a source's config. `source_root` is the folder being indexed (drives mode
    /// auto-detection, which reads the folder's contents); `config_path` is the centralized
    /// `~/.dora/sources/<name>/config.toml` where the file actually lives.
    pub fn load_or_default(source_root: &Path, config_path: &Path) -> Result<Self> {
        let raw: RawConfig = if config_path.exists() {
            let text = std::fs::read_to_string(config_path)?;
            toml::from_str(&text)?
        } else {
            RawConfig::default()
        };
        Ok(Self::resolve(raw, source_root))
    }

    fn resolve(raw: RawConfig, source_root: &Path) -> Self {
        let mode = Mode::resolve(&raw.source.mode, source_root);
        let chunk_d = mode.chunking_defaults();
        let embed_d = mode.embedder_defaults();
        let vault_d = mode.vault_defaults();

        let mut ignore = raw.vault.ignore.unwrap_or(vault_d.ignore);
        // Always force-include the dora/git/node_modules basics so users can't accidentally
        // wipe them by setting [vault] ignore = [].
        for required in [".dora", ".git", "node_modules"] {
            if !ignore.iter().any(|d| d == required) {
                ignore.push(required.to_string());
            }
        }

        let search_d = SearchConfig::default();

        Config {
            source: SourceConfig {
                mode: mode.as_str().to_string(),
            },
            vault: VaultConfig { ignore },
            chunking: ChunkingConfig {
                target_bytes: raw.chunking.target_bytes.unwrap_or(chunk_d.target_bytes),
                atomic_below_bytes: raw
                    .chunking
                    .atomic_below_bytes
                    .unwrap_or(chunk_d.atomic_below_bytes),
                overlap_bytes: raw.chunking.overlap_bytes.unwrap_or(chunk_d.overlap_bytes),
            },
            search: SearchConfig {
                top_k: raw.search.top_k.unwrap_or(search_d.top_k),
                collapse_per_file: raw
                    .search
                    .collapse_per_file
                    .unwrap_or(search_d.collapse_per_file),
            },
            embedder: EmbedderConfig {
                provider: raw.embedder.provider.unwrap_or(embed_d.provider),
                model: raw.embedder.model.unwrap_or(embed_d.model),
                api_key_env: raw.embedder.api_key_env.unwrap_or(embed_d.api_key_env),
                dimensions: raw.embedder.dimensions.or(embed_d.dimensions),
            },
            claude_code: {
                let d = ClaudeCodeConfig::default();
                ClaudeCodeConfig {
                    include_thinking: raw
                        .claude_code
                        .include_thinking
                        .unwrap_or(d.include_thinking),
                    settle_seconds: raw.claude_code.settle_seconds.unwrap_or(d.settle_seconds),
                }
            },
            codex: {
                let d = CodexConfig::default();
                CodexConfig {
                    include_reasoning: raw.codex.include_reasoning.unwrap_or(d.include_reasoning),
                    settle_seconds: raw.codex.settle_seconds.unwrap_or(d.settle_seconds),
                }
            },
            graph: GraphConfig {
                entities: raw.graph.entities.unwrap_or(false),
            },
            confidence: ConfidenceConfig {
                ann_floor: raw.confidence.ann_floor,
            },
        }
    }
}

impl Default for Config {
    /// Used by tests / call sites that want a config without touching the filesystem. Resolves
    /// against a non-existent path → mode defaults to `notes`.
    fn default() -> Self {
        Self::resolve(RawConfig::default(), Path::new("/__dora_default__"))
    }
}

impl Default for EmbedderConfig {
    fn default() -> Self {
        Self {
            provider: "fastembed".into(),
            model: "bge-base-en-v1.5-onnx-q".into(),
            api_key_env: "OPENAI_API_KEY".into(),
            dimensions: None,
        }
    }
}

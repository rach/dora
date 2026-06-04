use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use crate::chunk::Chunker;
use crate::config::Config;
use crate::embed::{self, DynEmbedder};
use crate::store::Store;

#[derive(Debug, Deserialize)]
struct EvalFile {
    #[serde(rename = "query")]
    queries: Vec<EvalQuery>,
}

#[derive(Debug, Deserialize, Clone)]
struct EvalQuery {
    name: String,
    query: String,
    relevant: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct EvalOutcome {
    name: String,
    query: String,
    relevant: Vec<String>,
    rank: Option<usize>,
    top_hit: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct EvalMetrics {
    queries: usize,
    r_at_1: f64,
    r_at_3: f64,
    r_at_5: f64,
    mrr: f64,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct EvalOptions {
    pub top_k: usize,
    pub min_r_at_1: Option<f64>,
    pub json: bool,
    pub disable_prf: bool,
    pub disable_graph: bool,
    pub compare_disable_graph: bool,
}

#[derive(Debug, Serialize)]
struct EvalReport {
    fixture: String,
    top_k: usize,
    disable_prf: bool,
    disable_graph: bool,
    metrics: EvalMetrics,
    outcomes: Vec<EvalOutcome>,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct EvalMetricDelta {
    r_at_1: f64,
    r_at_3: f64,
    r_at_5: f64,
    mrr: f64,
}

#[derive(Debug, Serialize)]
struct EvalCompareReport {
    graph_on: EvalReport,
    graph_off: EvalReport,
    delta: EvalMetricDelta,
}

pub(crate) fn cmd_eval(fixture: &Path, opts: EvalOptions) -> Result<()> {
    let fixture = fixture
        .canonicalize()
        .context("canonicalize eval fixture")?;
    let eval_file = load_queries(&fixture.join("queries.toml"))?;
    if eval_file.queries.is_empty() {
        bail!("eval fixture has no [[query]] entries");
    }
    if opts.top_k == 0 {
        bail!("--top-k must be at least 1");
    }
    if opts.compare_disable_graph && opts.disable_graph {
        bail!("--compare-disable-graph cannot be combined with --disable-graph");
    }

    let report = run_eval_report(
        &fixture,
        &eval_file.queries,
        opts.top_k,
        opts.disable_prf,
        opts.disable_graph,
    )?;

    if opts.compare_disable_graph {
        let graph_off = run_eval_report(
            &fixture,
            &eval_file.queries,
            opts.top_k,
            opts.disable_prf,
            true,
        )?;
        let delta = metric_delta(report.metrics, graph_off.metrics);
        let compare = EvalCompareReport {
            graph_on: report,
            graph_off,
            delta,
        };
        if opts.json {
            println!("{}", serde_json::to_string_pretty(&compare)?);
        } else {
            print_report(&compare.graph_on, opts.disable_prf || opts.disable_graph);
            println!("graph comparison:");
            println!(
                "  graph-off R@1 {:.3} R@3 {:.3} R@5 {:.3} MRR {:.3}",
                compare.graph_off.metrics.r_at_1,
                compare.graph_off.metrics.r_at_3,
                compare.graph_off.metrics.r_at_5,
                compare.graph_off.metrics.mrr
            );
            println!(
                "  delta     R@1 {:+.3} R@3 {:+.3} R@5 {:+.3} MRR {:+.3}",
                compare.delta.r_at_1, compare.delta.r_at_3, compare.delta.r_at_5, compare.delta.mrr
            );
        }
        if compare.delta.r_at_5 <= 0.0 || compare.delta.mrr <= 0.0 {
            bail!(
                "graph comparison failed: expected graph-on to beat graph-off on R@5 and MRR \
                 (delta R@5 {:+.3}, MRR {:+.3})",
                compare.delta.r_at_5,
                compare.delta.mrr
            );
        }
        if let Some(min) = opts.min_r_at_1 {
            if compare.graph_on.metrics.r_at_1 < min {
                bail!(
                    "R@1 {:.3} below required threshold {:.3}",
                    compare.graph_on.metrics.r_at_1,
                    min
                );
            }
        }
    } else {
        if opts.json {
            println!("{}", serde_json::to_string_pretty(&report)?);
        } else {
            print_report(&report, opts.disable_prf || opts.disable_graph);
        }
        if let Some(min) = opts.min_r_at_1 {
            if report.metrics.r_at_1 < min {
                bail!(
                    "R@1 {:.3} below required threshold {:.3}",
                    report.metrics.r_at_1,
                    min
                );
            }
        }
    }

    Ok(())
}

fn run_eval_report(
    fixture: &Path,
    queries: &[EvalQuery],
    top_k: usize,
    disable_prf: bool,
    disable_graph: bool,
) -> Result<EvalReport> {
    let temp_root = make_temp_root()?;
    let _env = EvalEnv::apply(disable_prf, disable_graph);
    let result = run_eval_in_temp(fixture, &temp_root, queries, top_k);
    let _ = fs::remove_dir_all(&temp_root);
    let (metrics, outcomes) = result?;
    Ok(EvalReport {
        fixture: fixture.display().to_string(),
        top_k,
        disable_prf,
        disable_graph,
        metrics,
        outcomes,
    })
}

fn print_report(report: &EvalReport, show_ablations: bool) {
    println!("fixture: {}", report.fixture);
    println!("top_k: {}", report.top_k);
    if show_ablations {
        println!(
            "ablations: prf={} graph={}",
            if report.disable_prf { "off" } else { "on" },
            if report.disable_graph { "off" } else { "on" }
        );
    }
    println!("queries: {}", report.metrics.queries);
    println!("R@1: {:.3}", report.metrics.r_at_1);
    println!("R@3: {:.3}", report.metrics.r_at_3);
    println!("R@5: {:.3}", report.metrics.r_at_5);
    println!("MRR: {:.3}", report.metrics.mrr);

    let failures: Vec<&EvalOutcome> = report
        .outcomes
        .iter()
        .filter(|o| o.rank.is_none())
        .collect();
    if !failures.is_empty() {
        println!("misses:");
        for failure in failures {
            println!("  {} (top: {:?})", failure.name, failure.top_hit);
        }
    }
}

fn metric_delta(on: EvalMetrics, off: EvalMetrics) -> EvalMetricDelta {
    EvalMetricDelta {
        r_at_1: on.r_at_1 - off.r_at_1,
        r_at_3: on.r_at_3 - off.r_at_3,
        r_at_5: on.r_at_5 - off.r_at_5,
        mrr: on.mrr - off.mrr,
    }
}

fn run_eval_in_temp(
    fixture: &Path,
    temp_root: &Path,
    queries: &[EvalQuery],
    top_k: usize,
) -> Result<(EvalMetrics, Vec<EvalOutcome>)> {
    copy_docs(&fixture.join("docs"), temp_root)?;
    // Eval deliberately keeps a co-located `.dora/` inside its ephemeral temp source root —
    // it's throwaway, never registered, never migrated, so it stays decoupled from the
    // production centralized resolver in paths.rs / main.rs.
    let eval_dora = temp_root.join(".dora");
    fs::create_dir_all(&eval_dora)?;

    let cfg = Config::load_or_default(temp_root, &eval_dora.join("config.toml"))
        .context("load eval config")?;
    let model_dir = eval_models_dir(temp_root)?;
    let embedder: DynEmbedder = embed::from_config(&cfg.embedder, &model_dir)?;
    let chunker: Box<dyn Chunker> = crate::chunk::from_config(&cfg, temp_root);
    let db = eval_dora.join("index.db");
    if db.exists() && !crate::meta_matches(&db, embedder.as_ref())? {
        fs::remove_file(&db).context("remove stale eval index")?;
    }
    let mut store = Store::open(&db, embedder.dims())?;
    crate::write_identity_meta(&store, embedder.as_ref())?;
    crate::run_incremental_index(
        temp_root,
        &cfg,
        chunker.as_ref(),
        embedder.as_ref(),
        &mut store,
        false,
    )?;

    let mut outcomes = Vec::with_capacity(queries.len());
    for q in queries {
        let hits = crate::search::search_with_confidence(
            &q.query,
            &store,
            embedder.as_ref(),
            temp_root,
            "eval",
            crate::search::SearchOptions {
                top_k,
                diagnostics: true,
                ..Default::default()
            },
        )?
        .hits;
        let relevant: HashSet<&str> = q.relevant.iter().map(String::as_str).collect();
        let rank = hits
            .iter()
            .enumerate()
            .find(|(_, hit)| relevant.contains(hit.path.as_str()))
            .map(|(idx, _)| idx + 1);
        let top_hit = hits.first().map(|h| h.path.clone());
        outcomes.push(EvalOutcome {
            name: q.name.clone(),
            query: q.query.clone(),
            relevant: q.relevant.clone(),
            rank,
            top_hit,
        });
    }

    Ok((compute_metrics(&outcomes), outcomes))
}

struct EvalEnv {
    prf: Option<String>,
    graph: Option<String>,
}

impl EvalEnv {
    fn apply(disable_prf: bool, disable_graph: bool) -> Self {
        let env = Self {
            prf: std::env::var("DORA_DISABLE_PRF").ok(),
            graph: std::env::var("DORA_DISABLE_GRAPH").ok(),
        };
        if disable_prf {
            std::env::set_var("DORA_DISABLE_PRF", "1");
        }
        if disable_graph {
            std::env::set_var("DORA_DISABLE_GRAPH", "1");
        }
        env
    }
}

impl Drop for EvalEnv {
    fn drop(&mut self) {
        match &self.prf {
            Some(v) => std::env::set_var("DORA_DISABLE_PRF", v),
            None => std::env::remove_var("DORA_DISABLE_PRF"),
        }
        match &self.graph {
            Some(v) => std::env::set_var("DORA_DISABLE_GRAPH", v),
            None => std::env::remove_var("DORA_DISABLE_GRAPH"),
        }
    }
}

fn load_queries(path: &Path) -> Result<EvalFile> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    toml::from_str(&text).with_context(|| format!("parse {}", path.display()))
}

fn copy_docs(from: &Path, to: &Path) -> Result<()> {
    if !from.is_dir() {
        bail!("eval fixture missing docs/ at {}", from.display());
    }
    for entry in walkdir::WalkDir::new(from) {
        let entry = entry?;
        if entry.file_type().is_dir() {
            continue;
        }
        let rel = entry.path().strip_prefix(from)?;
        let dst = to.join(rel);
        if let Some(parent) = dst.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(entry.path(), &dst)
            .with_context(|| format!("copy {} -> {}", entry.path().display(), dst.display()))?;
    }
    Ok(())
}

fn make_temp_root() -> Result<PathBuf> {
    let mut root = std::env::temp_dir();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    root.push(format!("dora-eval-{}-{nonce}", std::process::id()));
    fs::create_dir_all(&root)?;
    Ok(root)
}

fn eval_models_dir(fallback_root: &Path) -> Result<PathBuf> {
    let dir = dirs::cache_dir()
        .map(|p| p.join("dora").join("eval-models"))
        .unwrap_or_else(|| fallback_root.join(".dora").join("models"));
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

fn compute_metrics(outcomes: &[EvalOutcome]) -> EvalMetrics {
    let n = outcomes.len().max(1) as f64;
    let recall_at = |k: usize| -> f64 {
        outcomes
            .iter()
            .filter(|o| o.rank.map(|r| r <= k).unwrap_or(false))
            .count() as f64
            / n
    };
    let mrr = outcomes
        .iter()
        .map(|o| o.rank.map(|r| 1.0 / r as f64).unwrap_or(0.0))
        .sum::<f64>()
        / n;
    EvalMetrics {
        queries: outcomes.len(),
        r_at_1: recall_at(1),
        r_at_3: recall_at(3),
        r_at_5: recall_at(5),
        mrr,
    }
}

#[cfg(test)]
mod tests {
    use super::{compute_metrics, EvalOutcome};

    #[test]
    fn metrics_compute_recall_and_mrr() {
        let outcomes = vec![
            EvalOutcome {
                name: "first".to_string(),
                query: "first query".to_string(),
                relevant: vec!["a.md".to_string()],
                rank: Some(1),
                top_hit: Some("a.md".to_string()),
            },
            EvalOutcome {
                name: "third".to_string(),
                query: "third query".to_string(),
                relevant: vec!["b.md".to_string()],
                rank: Some(3),
                top_hit: Some("x.md".to_string()),
            },
            EvalOutcome {
                name: "miss".to_string(),
                query: "miss query".to_string(),
                relevant: vec!["c.md".to_string()],
                rank: None,
                top_hit: Some("z.md".to_string()),
            },
        ];
        let metrics = compute_metrics(&outcomes);
        assert_eq!(metrics.queries, 3);
        assert!((metrics.r_at_1 - (1.0 / 3.0)).abs() < 1e-9);
        assert!((metrics.r_at_3 - (2.0 / 3.0)).abs() < 1e-9);
        assert!((metrics.r_at_5 - (2.0 / 3.0)).abs() < 1e-9);
        assert!((metrics.mrr - ((1.0 + 1.0 / 3.0) / 3.0)).abs() < 1e-9);
    }
}

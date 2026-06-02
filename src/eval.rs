use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use crate::chunk::Chunker;
use crate::config::Config;
use crate::embed::{self, DynEmbedder};
use crate::store::Store;

const DEFAULT_TOP_K: usize = 5;

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

#[derive(Debug, Clone)]
struct EvalOutcome {
    name: String,
    rank: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
struct EvalMetrics {
    queries: usize,
    r_at_1: f64,
    r_at_3: f64,
    r_at_5: f64,
    mrr: f64,
}

pub(crate) fn cmd_eval(fixture: &Path, min_r_at_1: Option<f64>) -> Result<()> {
    let fixture = fixture
        .canonicalize()
        .context("canonicalize eval fixture")?;
    let eval_file = load_queries(&fixture.join("queries.toml"))?;
    if eval_file.queries.is_empty() {
        bail!("eval fixture has no [[query]] entries");
    }

    let temp_root = make_temp_root()?;
    let result = run_eval_in_temp(&fixture, &temp_root, &eval_file.queries);
    let _ = fs::remove_dir_all(&temp_root);
    let (metrics, outcomes) = result?;

    println!("queries: {}", metrics.queries);
    println!("R@1: {:.3}", metrics.r_at_1);
    println!("R@3: {:.3}", metrics.r_at_3);
    println!("R@5: {:.3}", metrics.r_at_5);
    println!("MRR: {:.3}", metrics.mrr);

    let failures: Vec<&EvalOutcome> = outcomes.iter().filter(|o| o.rank.is_none()).collect();
    if !failures.is_empty() {
        println!("misses:");
        for failure in failures {
            println!("  {}", failure.name);
        }
    }

    if let Some(min) = min_r_at_1 {
        if metrics.r_at_1 < min {
            bail!(
                "R@1 {:.3} below required threshold {:.3}",
                metrics.r_at_1,
                min
            );
        }
    }

    Ok(())
}

fn run_eval_in_temp(
    fixture: &Path,
    temp_root: &Path,
    queries: &[EvalQuery],
) -> Result<(EvalMetrics, Vec<EvalOutcome>)> {
    copy_docs(&fixture.join("docs"), temp_root)?;
    fs::create_dir_all(crate::dora_dir(temp_root))?;

    let cfg = Config::load_or_default(temp_root).context("load eval config")?;
    let embedder: DynEmbedder = embed::from_config(&cfg.embedder, &crate::models_dir(temp_root))?;
    let chunker: Box<dyn Chunker> = crate::chunk::from_config(&cfg, temp_root);
    let db = crate::db_path(temp_root);
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
        let hits = crate::search::search(
            &q.query,
            &store,
            embedder.as_ref(),
            temp_root,
            "eval",
            crate::search::SearchOptions {
                top_k: DEFAULT_TOP_K,
                diagnostics: true,
                ..Default::default()
            },
        )?;
        let relevant: HashSet<&str> = q.relevant.iter().map(String::as_str).collect();
        let rank = hits
            .iter()
            .enumerate()
            .find(|(_, hit)| relevant.contains(hit.path.as_str()))
            .map(|(idx, _)| idx + 1);
        outcomes.push(EvalOutcome {
            name: q.name.clone(),
            rank,
        });
    }

    Ok((compute_metrics(&outcomes), outcomes))
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
                rank: Some(1),
            },
            EvalOutcome {
                name: "third".to_string(),
                rank: Some(3),
            },
            EvalOutcome {
                name: "miss".to_string(),
                rank: None,
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

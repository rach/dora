//! Hybrid search: FTS5 + sqlite-vec ANN merged with Reciprocal Rank Fusion.

use anyhow::Result;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::embed::Embedder;
use crate::store::Store;

// search takes `&dyn Embedder` so it works with both FastembedEmbedder and OpenAIEmbedder
// (or any future provider) without monomorphizing per impl.

const RRF_K: f64 = 60.0;
const PER_LIST_LIMIT: usize = 50;

#[derive(Debug, Serialize)]
pub struct Hit {
    /// Name of the registered source this hit came from. Set to the source's registry name
    /// in multi-source MCP mode, or to the cwd basename / `--source` argument otherwise.
    pub source: String,
    pub path: String,
    pub line: usize,
    pub score: f64,
    pub heading_path: String,
    pub snippet: String,
    /// Optional user-defined context for this path's subtree, set via `dora context add`.
    /// Surfaces to agents so e.g. `/api` hits can carry "REST API reference" while `/sdk`
    /// hits carry "TypeScript SDK reference" — qmd parity.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
}

/// Knobs threaded through the public `search()` entry point. Callers (CLI, MCP) build
/// this from their flag set. `Default` matches v0.3.x behavior (top_k=10, chunks, no
/// path filter).
#[derive(Debug, Clone)]
pub struct SearchOptions<'a> {
    pub top_k: usize,
    /// When `Some`, drop any hit whose RRF score is below the threshold.
    pub min_score: Option<f64>,
    /// When true, ignore `top_k` — return every hit that passed `min_score` (if set).
    pub all: bool,
    pub path_prefix: Option<&'a str>,
    pub output: OutputMode,
}

impl<'a> Default for SearchOptions<'a> {
    fn default() -> Self {
        Self {
            top_k: 10,
            min_score: None,
            all: false,
            path_prefix: None,
            output: OutputMode::Chunks,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OutputMode {
    #[default]
    Chunks,
    /// Dedupe results by file path, keeping the max-score chunk per file. Snippet /
    /// heading_path / line carried over from the best chunk; line=0 + empty fields if the
    /// caller wants paths only.
    Files,
}

pub fn search(
    query: &str,
    store: &Store,
    embedder: &dyn Embedder,
    source_root: &Path,
    source_name: &str,
    opts: SearchOptions<'_>,
) -> Result<Vec<Hit>> {
    let query_vec = embedder.embed_one(query)?;
    let ann = store.search_ann(&query_vec, PER_LIST_LIMIT, opts.path_prefix)?;

    let fts_query = safe_fts_query(query);
    let fts = if fts_query.is_empty() {
        Vec::new()
    } else {
        store
            .search_fts(&fts_query, PER_LIST_LIMIT, opts.path_prefix)
            .unwrap_or_else(|err| {
                eprintln!("fts query failed (continuing with ANN only): {err}");
                Vec::new()
            })
    };

    // Third arm: literal substring scan. Closes the gaps where FTS5's tokenizer doesn't
    // help — camelCase identifiers (`processRequest`), snake_case fragments (`foo_bar`),
    // magic constants (`E_NOENT`). For natural-language queries this typically returns
    // nothing (or noise that RRF discounts to the bottom), so it never hurts.
    let trimmed_query = query.trim();
    let literal = if trimmed_query.is_empty() {
        Vec::new()
    } else {
        store
            .search_literal(trimmed_query, PER_LIST_LIMIT, opts.path_prefix)
            .unwrap_or_else(|err| {
                eprintln!("literal query failed (continuing without it): {err}");
                Vec::new()
            })
    };

    let merged = rrf_merge_n(&[&fts, &ann, &literal]);
    // Per-file collapse: best chunk per file. When `all` is set we keep every collapsed
    // hit and let `min_score` do the filtering downstream; otherwise we still cap at top_k.
    let collapse_cap = if opts.all { merged.len() } else { opts.top_k };
    let merged = collapse_per_file(&merged, store, collapse_cap)?;

    // Apply min_score filter, if any.
    let merged: Vec<(i64, f64)> = match opts.min_score {
        Some(threshold) => merged
            .into_iter()
            .filter(|(_, score)| *score >= threshold)
            .collect(),
        None => merged,
    };

    // Apply top_k unless --all was set.
    let merged: Vec<(i64, f64)> = if opts.all {
        merged
    } else {
        merged.into_iter().take(opts.top_k).collect()
    };

    let mut hits = Vec::with_capacity(merged.len());
    for (chunk_id, score) in merged.into_iter() {
        if let Some(chunk) = store.fetch_chunk(chunk_id)? {
            let line = line_for_byte(source_root, Path::new(&chunk.path), chunk.start_byte);
            let context = store.context_for(&chunk.path).ok().flatten();
            hits.push(Hit {
                source: source_name.to_string(),
                path: chunk.path,
                line,
                score,
                heading_path: chunk.heading_path,
                snippet: snippet_from(&chunk.content),
                context,
            });
        }
    }

    // Files-mode: collapse already deduplicates by file, so we can just strip the
    // chunk-only fields here. (We could also skip the snippet_from work above, but
    // it's cheap and keeps the code path uniform.)
    if opts.output == OutputMode::Files {
        for h in hits.iter_mut() {
            h.line = 0;
            h.heading_path.clear();
            h.snippet.clear();
        }
    }

    Ok(hits)
}

fn collapse_per_file(
    merged: &[(i64, f64)],
    store: &Store,
    top_k: usize,
) -> Result<Vec<(i64, f64)>> {
    if merged.is_empty() {
        return Ok(Vec::new());
    }
    let ids: Vec<i64> = merged.iter().map(|(id, _)| *id).collect();
    let file_of = store.file_ids_for_chunks(&ids)?;
    let mut seen: HashSet<i64> = HashSet::new();
    let mut out: Vec<(i64, f64)> = Vec::with_capacity(top_k);
    for (chunk_id, score) in merged {
        if let Some(file_id) = file_of.get(chunk_id) {
            if seen.insert(*file_id) {
                out.push((*chunk_id, *score));
                if out.len() >= top_k {
                    break;
                }
            }
        }
    }
    Ok(out)
}

/// Reciprocal Rank Fusion across any number of ranked lists. Each list contributes
/// `1 / (RRF_K + rank)` per occurrence; the top of each list also earns a position bonus
/// (+0.05 for rank-1, +0.02 for rank-2/3) to sharpen precision. Final list is sorted by
/// summed score, descending. Bonus values match qmd's published fusion (`tobi/qmd`).
fn rrf_merge_n(lists: &[&[i64]]) -> Vec<(i64, f64)> {
    let mut scores: HashMap<i64, f64> = HashMap::new();
    for list in lists {
        for (rank, id) in list.iter().enumerate() {
            let base = 1.0 / (RRF_K + (rank + 1) as f64);
            let bonus = match rank {
                0 => 0.05,
                1 | 2 => 0.02,
                _ => 0.0,
            };
            *scores.entry(*id).or_insert(0.0) += base + bonus;
        }
    }
    let mut v: Vec<(i64, f64)> = scores.into_iter().collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    v
}

#[cfg(test)]
mod tests {
    use super::rrf_merge_n;

    #[test]
    fn rank_one_in_any_list_gets_bonus() {
        // Two lists, same IDs at different ranks. IDs 1 and 4 are #1 in different lists.
        let a: Vec<i64> = vec![1, 2, 3];
        let b: Vec<i64> = vec![4, 2, 5];
        let merged = rrf_merge_n(&[&a, &b]);
        // ID 1 is #1 in list A → base + 0.05. Same for ID 4 in list B.
        let score_of = |id: i64| merged.iter().find(|(i, _)| *i == id).unwrap().1;
        // The rank-1 IDs (1, 4) each get the +0.05 bonus on their single appearance.
        // Without the bonus they'd score 1/61 ≈ 0.0164; with bonus, ≈ 0.0664.
        let one = score_of(1);
        let four = score_of(4);
        assert!(one > 0.06, "rank-1 ID should get the +0.05 bonus, got {one}");
        assert!(four > 0.06, "rank-1 ID should get the +0.05 bonus, got {four}");
        // ID 2 appears at rank-2 in BOTH lists → 2 × (1/62 + 0.02) ≈ 0.0723. Should be top.
        let two = score_of(2);
        assert!(two > one, "rank-2 in both lists should beat single rank-1");
    }
}

/// Build an FTS5 query string that won't choke on natural-language punctuation.
/// Strategy: extract alphanumeric/`-`/`_` tokens, phrase-quote each, OR-join.
/// Empty result → caller skips FTS and falls back to ANN only.
fn safe_fts_query(q: &str) -> String {
    let tokens: Vec<String> = q
        .split_whitespace()
        .filter_map(|t| {
            let cleaned: String = t
                .chars()
                .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
                .collect();
            if cleaned.is_empty() {
                None
            } else {
                Some(format!("\"{cleaned}\""))
            }
        })
        .collect();
    if tokens.is_empty() {
        String::new()
    } else {
        tokens.join(" OR ")
    }
}

fn snippet_from(content: &str) -> String {
    let body = strip_leading_frontmatter(content);
    let first = body
        .lines()
        .find(|l| {
            let t = l.trim();
            !t.is_empty() && t != "---"
        })
        .unwrap_or("")
        .trim();
    let max_chars = 140;
    if first.chars().count() > max_chars {
        let truncated: String = first.chars().take(max_chars).collect();
        format!("{truncated}…")
    } else {
        first.to_string()
    }
}

/// If the chunk begins with `---\n…\n---` YAML frontmatter, return the slice past it.
/// Otherwise return the input unchanged.
fn strip_leading_frontmatter(content: &str) -> &str {
    let trimmed = content.trim_start_matches('\u{feff}'); // tolerate stray BOM
    let mut lines = trimmed.lines();
    let Some(first) = lines.next() else { return content; };
    if first.trim() != "---" {
        return content;
    }
    // Scan for the closing `---` line.
    let mut consumed = first.len() + 1; // +1 for the newline
    for line in lines {
        consumed += line.len() + 1;
        if line.trim() == "---" {
            return &trimmed[consumed.min(trimmed.len())..];
        }
    }
    // No closing fence — leave content as-is.
    content
}

fn line_for_byte(vault_root: &Path, rel_path: &Path, start_byte: usize) -> usize {
    let full: PathBuf = vault_root.join(rel_path);
    let Ok(bytes) = std::fs::read(&full) else {
        return 1;
    };
    let cap = start_byte.min(bytes.len());
    bytes[..cap].iter().filter(|b| **b == b'\n').count() + 1
}

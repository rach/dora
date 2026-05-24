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
}

pub fn search(
    query: &str,
    store: &Store,
    embedder: &dyn Embedder,
    source_root: &Path,
    source_name: &str,
    top_k: usize,
    path_prefix: Option<&str>,
) -> Result<Vec<Hit>> {
    let query_vec = embedder.embed_one(query)?;
    let ann = store.search_ann(&query_vec, PER_LIST_LIMIT, path_prefix)?;

    let fts_query = safe_fts_query(query);
    let fts = if fts_query.is_empty() {
        Vec::new()
    } else {
        store
            .search_fts(&fts_query, PER_LIST_LIMIT, path_prefix)
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
            .search_literal(trimmed_query, PER_LIST_LIMIT, path_prefix)
            .unwrap_or_else(|err| {
                eprintln!("literal query failed (continuing without it): {err}");
                Vec::new()
            })
    };

    let merged = rrf_merge_n(&[&fts, &ann, &literal]);
    // Per-file collapse: best chunk per file (default behavior for v0 OSS).
    let merged = collapse_per_file(&merged, store, top_k)?;

    let mut hits = Vec::with_capacity(top_k);
    for (chunk_id, score) in merged.into_iter() {
        if let Some(chunk) = store.fetch_chunk(chunk_id)? {
            let line = line_for_byte(source_root, Path::new(&chunk.path), chunk.start_byte);
            hits.push(Hit {
                source: source_name.to_string(),
                path: chunk.path,
                line,
                score,
                heading_path: chunk.heading_path,
                snippet: snippet_from(&chunk.content),
            });
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
/// `1 / (RRF_K + rank)` per occurrence; final list is sorted by summed score, descending.
fn rrf_merge_n(lists: &[&[i64]]) -> Vec<(i64, f64)> {
    let mut scores: HashMap<i64, f64> = HashMap::new();
    for list in lists {
        for (rank, id) in list.iter().enumerate() {
            *scores.entry(*id).or_insert(0.0) += 1.0 / (RRF_K + (rank + 1) as f64);
        }
    }
    let mut v: Vec<(i64, f64)> = scores.into_iter().collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    v
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

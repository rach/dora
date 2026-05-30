//! Hybrid search: FTS5 + sqlite-vec ANN + literal substring + pseudo-relevance feedback,
//! merged with Reciprocal Rank Fusion.

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

/// v0.10 graph-PPR retrieval (Layer C). Seed PPR with the top-`GRAPH_SEED_TOPK` RRF hits;
/// add at most `GRAPH_BOOST_CAP` to a candidate's score, scaled by its normalized PPR mass.
/// The cap is tie-breaker-scale (RRF top scores are ~0.05–0.15 and the rank bonuses are
/// 0.02–0.05): the graph nudges ordering among near-tied hits and promotes graph-central
/// candidates, but can't bulldoze a strong lexical/vector match. Keeps the boost net-positive
/// on associative/linked corpora without regressing single-hop retrieval on flat ones.
const GRAPH_SEED_TOPK: usize = 10;
const GRAPH_BOOST_CAP: f64 = 0.03;
const GRAPH_PPR_ITERATIONS: usize = 30;
const GRAPH_PPR_DAMPING: f64 = 0.85;

/// v0.6 PRF (pseudo-relevance feedback) constants. Tuned blind; the eval harness drives
/// further tuning if it matters. The vector ANN arm is presumed to be the best proxy for
/// "what the corpus is talking about for this query", so we mine its top hits for
/// vocabulary that overlaps with the corpus but doesn't yet appear in the query.
const PRF_ANN_TOP: usize = 10;
const PRF_MAX_TERMS: usize = 5;
const PRF_MIN_TERM_LEN: usize = 3;

#[derive(Debug, Serialize)]
pub struct Hit {
    /// Stable id of the chunk in this source's index. Exposed so the MCP layer (or agents
    /// running follow-up tools) can attribute reads back to the search that returned them,
    /// and so the v0.6 `usage` log can carry the actual chunk IDs (not paths) for v0.7's
    /// signal-based reranker.
    pub chunk_id: i64,
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
    /// Boolean intersection terms (`--and` on CLI, `and` on the MCP `search` tool). A chunk
    /// must score above zero on the primary AND every and-query to remain a candidate; final
    /// score is the harmonic mean of the per-query normalized scores. Empty = no intersection.
    pub and_queries: Vec<String>,
    /// Boolean exclusion terms (`--not` / `not`). Chunks scoring highly on any not-query are
    /// dropped; weaker matches get a soft demote. Empty = no exclusion.
    pub not_queries: Vec<String>,
}

impl<'a> Default for SearchOptions<'a> {
    fn default() -> Self {
        Self {
            top_k: 10,
            min_score: None,
            all: false,
            path_prefix: None,
            output: OutputMode::Chunks,
            and_queries: Vec::new(),
            not_queries: Vec::new(),
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
    // Primary hybrid pass: 4-arm RRF over the user's query. `query_vec` is kept for
    // usage logging downstream.
    let (mut merged, query_vec) = compute_merged(query, store, embedder, opts.path_prefix)?;

    // Layer C: Personalized-PageRank graph boost. Seed PPR with the current top hits
    // (weighted by RRF score), spread across the document graph (wikilinks + derived
    // edges), and add a bounded boost so chunks densely connected to the seed surface —
    // associative recall without an LLM. Applied to the FULL merged list (before collapse)
    // so a graph-central chunk can climb into top_k, not just reorder the already-top-k.
    // Composes additively with the planned v0.7 usage boost. Gated on the source having a
    // graph; `DORA_DISABLE_GRAPH=1` is an eval-only A/B switch. Primary-query only — side
    // queries (--and / --not) don't propagate graph drift.
    let graph_enabled = std::env::var("DORA_DISABLE_GRAPH")
        .map(|v| v != "1")
        .unwrap_or(true);
    if graph_enabled && !merged.is_empty() {
        apply_graph_boost(store, &mut merged);
    }

    // Boolean overlay: --and intersects (harmonic mean of normalized scores), --not
    // hard-drops + soft-demotes. Each side query runs the same compute_merged pipeline
    // (no graph boost, by design). No-op when both lists are empty — backward-compatible.
    let merged = if opts.and_queries.is_empty() && opts.not_queries.is_empty() {
        merged
    } else {
        apply_boolean(
            merged,
            &opts.and_queries,
            &opts.not_queries,
            store,
            embedder,
            opts.path_prefix,
        )?
    };

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
                chunk_id,
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

    // Best-effort usage logging. The MCP layer's ring buffer + `mark_used_by_query` patches
    // `used_chunk_id` if a follow-up `multi_get` reads one of the returned paths within 60s.
    // CLI-only invocations never get attributed, which is fine for v0.6 (data-collection).
    let chunk_ids: Vec<i64> = hits.iter().map(|h| h.chunk_id).collect();
    let json = serde_json::to_string(&chunk_ids).unwrap_or_default();
    if let Err(err) = store.log_usage(query, &query_vec, &json) {
        eprintln!("dora: usage logging failed (continuing): {err}");
    }

    Ok(hits)
}

/// One pass of dora's 4-arm hybrid pipeline (FTS + ANN + literal + PRF, fused via RRF).
/// Returns the merged ranked list **without** the Layer-C graph boost and **without**
/// per-file collapse / min_score / top_k truncation — those are the caller's concern. The
/// query's embedding is returned too so the caller can reuse it (usage logging, etc.)
/// without re-running the embedder.
///
/// Used by `search()` for the user's primary query and by `apply_boolean` for each `--and`
/// / `--not` side query, so all queries share identical ranking semantics.
fn compute_merged(
    query: &str,
    store: &Store,
    embedder: &dyn Embedder,
    path_prefix: Option<&str>,
) -> Result<(Vec<(i64, f64)>, Vec<f32>)> {
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

    let prf_enabled = std::env::var("DORA_DISABLE_PRF")
        .map(|v| v != "1")
        .unwrap_or(true);
    let prf_terms = if prf_enabled {
        compute_prf_terms(store, &ann, query, PRF_MAX_TERMS)
    } else {
        Vec::new()
    };
    let prf = if prf_terms.is_empty() {
        Vec::new()
    } else {
        let prf_query = prf_terms
            .iter()
            .map(|t| format!("\"{t}\""))
            .collect::<Vec<_>>()
            .join(" OR ");
        store
            .search_fts(&prf_query, PER_LIST_LIMIT, path_prefix)
            .unwrap_or_else(|err| {
                eprintln!("prf query failed (continuing without it): {err}");
                Vec::new()
            })
    };

    Ok((rrf_merge_n(&[&fts, &ann, &literal, &prf]), query_vec))
}

/// Personalized-PageRank boost over the document graph. Seeds PPR with the merged top hits
/// (weighted by RRF score), runs it over the source's wikilink + derived-edge graph, and adds
/// a bounded boost to each candidate by its normalized PPR mass. Re-ranks the candidate pool
/// in place. No-op (silent) when the source has no graph. Composes additively with future
/// boosts (e.g. v0.7 usage signal) since the boost is capped.
fn apply_graph_boost(store: &Store, merged: &mut Vec<(i64, f64)>) {
    let edges = match store.graph_edges_for_ppr() {
        Ok(e) if !e.is_empty() => e,
        _ => return,
    };
    let mut seed: HashMap<i64, f64> = HashMap::new();
    for (id, score) in merged.iter().take(GRAPH_SEED_TOPK) {
        seed.insert(*id, *score);
    }
    if seed.is_empty() {
        return;
    }
    let ppr = crate::pagerank::compute_ppr(
        &edges,
        &seed,
        GRAPH_PPR_ITERATIONS,
        GRAPH_PPR_DAMPING,
    );
    let max_ppr = ppr.values().copied().fold(0.0f64, f64::max);
    if max_ppr <= 0.0 {
        return;
    }
    for (id, score) in merged.iter_mut() {
        if let Some(mass) = ppr.get(id) {
            *score += GRAPH_BOOST_CAP * (mass / max_ppr);
        }
    }
    merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
}

/// `--not` thresholds. A candidate with not-score above `NOT_HARD_DROP` is removed entirely
/// ("it's really about Z"); softer matches get `NOT_ALPHA * not_score` subtracted from their
/// running score (gentle demote). Combined: "demote, and drop the obvious cases."
const NOT_HARD_DROP: f64 = 0.5;
const NOT_ALPHA: f64 = 0.5;

/// Boolean overlay on a hybrid-search result. For each `--and` query, runs the same
/// `compute_merged` pipeline, normalizes scores to [0,1], and intersects: a chunk must
/// appear in the primary AND every and-list to survive, with combined score = harmonic mean
/// across all lists (asymmetry is punished). For each `--not` query, runs the same pipeline,
/// drops candidates that score highly, and subtracts a fraction of softer matches' scores.
///
/// All side queries skip the Layer-C graph boost on purpose — the graph represents the
/// PRIMARY intent; we don't want exclusion or intersection terms to propagate through it.
fn apply_boolean(
    primary: Vec<(i64, f64)>,
    and_queries: &[String],
    not_queries: &[String],
    store: &Store,
    embedder: &dyn Embedder,
    path_prefix: Option<&str>,
) -> Result<Vec<(i64, f64)>> {
    let mut primary = primary;
    normalize_scores(&mut primary);

    // Build the per-and lookup tables (normalized).
    let and_maps: Vec<HashMap<i64, f64>> = and_queries
        .iter()
        .filter(|q| {
            let trimmed = q.trim().is_empty();
            if trimmed {
                eprintln!("dora: skipping empty --and term");
            }
            !trimmed
        })
        .map(|q| {
            let (mut list, _vec) = compute_merged(q, store, embedder, path_prefix)?;
            normalize_scores(&mut list);
            Ok(list.into_iter().collect())
        })
        .collect::<Result<_>>()?;

    // Intersection: drop candidates missing from any and-map; combine via harmonic mean.
    let mut out: Vec<(i64, f64)> = primary
        .into_iter()
        .filter_map(|(id, p_score)| {
            let mut scores = Vec::with_capacity(and_maps.len() + 1);
            scores.push(p_score);
            for m in &and_maps {
                match m.get(&id) {
                    Some(v) => scores.push(*v),
                    None => return None,
                }
            }
            Some((id, harmonic_mean(&scores)))
        })
        .collect();

    // Exclusion: per not-query, hard-drop above the threshold, soft-subtract below.
    for nq in not_queries.iter().filter(|q| {
        let trimmed = q.trim().is_empty();
        if trimmed {
            eprintln!("dora: skipping empty --not term");
        }
        !trimmed
    }) {
        let (mut list, _vec) = compute_merged(nq, store, embedder, path_prefix)?;
        normalize_scores(&mut list);
        let map: HashMap<i64, f64> = list.into_iter().collect();
        out.retain(|(id, _)| {
            map.get(id).map(|v| *v <= NOT_HARD_DROP).unwrap_or(true)
        });
        for (id, score) in out.iter_mut() {
            if let Some(v) = map.get(id) {
                *score -= NOT_ALPHA * v;
            }
        }
    }

    out.sort_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(out)
}

/// Divide every score by the max so the list is in [0,1]. No-op on empty / all-zero lists.
fn normalize_scores(list: &mut [(i64, f64)]) {
    let max = list.iter().map(|(_, s)| *s).fold(0.0f64, f64::max);
    if max > 0.0 {
        for (_, s) in list.iter_mut() {
            *s /= max;
        }
    }
}

/// Harmonic mean of positive scores. Punishes asymmetry: `harmonic_mean(&[0.9, 0.1])`
/// (≈ 0.18) is much lower than `harmonic_mean(&[0.5, 0.5])` (= 0.5). Returns 0 if any
/// element is ≤ 0 (a zero score in an intersection means "not a candidate," so the
/// combined score is zero).
fn harmonic_mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut denom = 0.0f64;
    for x in xs {
        if *x <= 0.0 {
            return 0.0;
        }
        denom += 1.0 / *x;
    }
    xs.len() as f64 / denom
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
    use super::{compute_prf_terms, rrf_merge_n};
    use crate::store::{ChunkRow, Store, init_sqlite_vec};
    use tempfile::tempdir;

    /// PRF must surface vocabulary-adjacent terms from the corpus while filtering out
    /// stopwords and any token already in the user's query.
    #[test]
    fn prf_terms_excludes_stopwords_and_query_words() {
        init_sqlite_vec();
        let dir = tempdir().unwrap();
        let db = dir.path().join("p.db");
        let mut store = Store::open(&db, 4).unwrap();

        // Seed two chunks talking about "lead scoring" / "GTM signals" — vocabulary the
        // user's "ICP tracking" query does NOT contain. We want PRF to surface those.
        let emb = vec![0.0f32, 0.0, 0.0, 0.0];
        let rows = vec![
            ChunkRow {
                idx: 0,
                heading_path: "",
                content: "Lead scoring requires a clear ICP model. The signals are: \
                          firmographic, demographic, behavioral. Scoring is iterative.",
                start_byte: 0,
                end_byte: 200,
                embedding: &emb,
                kind: "prose",
                symbol: None,
                parent_chunk_idx: None,
            },
            ChunkRow {
                idx: 1,
                heading_path: "",
                content: "GTM motion: outbound prospecting against your ICP. Use \
                          firmographic signals to prioritize accounts. Scoring matters.",
                start_byte: 200,
                end_byte: 400,
                embedding: &emb,
                kind: "prose",
                symbol: None,
                parent_chunk_idx: None,
            },
        ];
        store
            .upsert_file_with_chunks("notes.md", 1, 400, "hash", &rows, &[])
            .unwrap();

        let ann_ids = vec![1i64, 2i64];
        let query = "how to track ICP";
        let terms = compute_prf_terms(&store, &ann_ids, query, 5);

        // ICP is in the query → excluded. Stopwords ("the", "to", "your") → excluded.
        for t in &terms {
            assert_ne!(t, "icp", "query word must not be in PRF expansion");
            assert_ne!(t, "the", "stopword must not be in PRF expansion");
            assert_ne!(t, "to", "stopword must not be in PRF expansion");
        }
        // "scoring" appears 3× across both chunks → should be a top expansion term.
        assert!(
            terms.iter().any(|t| t == "scoring"),
            "expected 'scoring' in PRF terms, got {terms:?}"
        );
        // "signals" appears 2× and isn't a stopword → expect it too.
        assert!(
            terms.iter().any(|t| t == "signals"),
            "expected 'signals' in PRF terms, got {terms:?}"
        );
        assert!(terms.len() <= 5, "PRF respects max-terms cap");
    }

    #[test]
    fn prf_terms_empty_when_no_ann_hits() {
        init_sqlite_vec();
        let dir = tempdir().unwrap();
        let db = dir.path().join("p2.db");
        let store = Store::open(&db, 4).unwrap();
        let terms = compute_prf_terms(&store, &[], "anything", 5);
        assert!(terms.is_empty(), "no ANN hits → no PRF terms");
    }

    #[test]
    fn harmonic_mean_punishes_asymmetry() {
        // Two chunks with the same arithmetic mean but very different distributions.
        let asymmetric = super::harmonic_mean(&[0.9, 0.1]);
        let symmetric = super::harmonic_mean(&[0.5, 0.5]);
        assert!(
            asymmetric < symmetric,
            "expected harmonic_mean(0.9,0.1)={asymmetric} < harmonic_mean(0.5,0.5)={symmetric}"
        );
        // Hard zero short-circuits to zero (intersection: a missing list zeroes the chunk).
        assert_eq!(super::harmonic_mean(&[0.5, 0.0, 0.5]), 0.0);
        // Single-element list returns that element.
        assert!((super::harmonic_mean(&[0.42]) - 0.42).abs() < 1e-9);
    }

    #[test]
    fn normalize_scores_to_unit_interval() {
        let mut list = vec![(1, 4.0), (2, 2.0), (3, 1.0)];
        super::normalize_scores(&mut list);
        assert_eq!(list[0].1, 1.0);
        assert_eq!(list[1].1, 0.5);
        assert_eq!(list[2].1, 0.25);
        // No-op on empty + all-zero lists.
        let mut empty: Vec<(i64, f64)> = Vec::new();
        super::normalize_scores(&mut empty);
        assert!(empty.is_empty());
        let mut zeros = vec![(1, 0.0), (2, 0.0)];
        super::normalize_scores(&mut zeros);
        assert_eq!(zeros, vec![(1, 0.0), (2, 0.0)]);
    }

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

/// Pseudo-relevance feedback term mining. Pulls the top vector-ANN chunks (the embedder's
/// best guess at "what's the corpus talking about for this query"), word-counts their
/// content, drops stopwords + query-words + short tokens, and returns the top-`max` by
/// frequency. The result is a small set of corpus vocabulary likely to be relevant — fed
/// into FTS as a fourth `rrf_merge_n` arm to close vocabulary gaps without an LLM.
fn compute_prf_terms(store: &Store, ann_ids: &[i64], query: &str, max: usize) -> Vec<String> {
    let top = ann_ids
        .iter()
        .take(PRF_ANN_TOP)
        .copied()
        .collect::<Vec<_>>();
    if top.is_empty() || max == 0 {
        return Vec::new();
    }
    let query_words: HashSet<String> = query
        .to_lowercase()
        .split_whitespace()
        .map(|s| {
            s.trim_matches(|c: char| !c.is_alphanumeric())
                .to_string()
        })
        .filter(|s| !s.is_empty())
        .collect();
    let mut freq: HashMap<String, usize> = HashMap::new();
    for id in &top {
        let Some(chunk) = store.fetch_chunk(*id).ok().flatten() else {
            continue;
        };
        for raw in chunk.content.split_whitespace() {
            let w: String = raw
                .to_lowercase()
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_string();
            if w.len() < PRF_MIN_TERM_LEN
                || query_words.contains(&w)
                || STOPWORDS.contains(&w.as_str())
                || !w.chars().any(|c| c.is_alphabetic())
            {
                continue;
            }
            *freq.entry(w).or_insert(0) += 1;
        }
    }
    let mut ranked: Vec<(String, usize)> = freq.into_iter().collect();
    // Sort by frequency desc, then alphabetic asc for deterministic results when frequencies tie.
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked.into_iter().take(max).map(|(w, _)| w).collect()
}

/// English stopwords for PRF. ~200 of the most-frequent function words — enough to keep PRF
/// from regressing to "the, and, of" while staying small enough to compile in. The corpus
/// is overwhelmingly English in practice; if non-English corpora become a thing we'll add
/// per-language lists keyed off the source's configured language.
pub(crate) const STOPWORDS: &[&str] = &[
    "a", "about", "above", "after", "again", "against", "all", "also", "am", "an", "and",
    "any", "are", "aren", "as", "at", "back", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "come", "could", "couldn", "day",
    "did", "didn", "do", "does", "doesn", "doing", "don", "down", "during", "each",
    "either", "end", "even", "every", "few", "first", "for", "from", "further", "get",
    "give", "go", "going", "good", "got", "had", "hadn", "has", "hasn", "have", "haven",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
    "however", "i", "if", "in", "into", "is", "isn", "it", "its", "itself", "just",
    "know", "last", "left", "let", "like", "look", "made", "make", "making", "many",
    "may", "me", "might", "more", "most", "mustn", "my", "myself", "need", "needs",
    "new", "no", "nor", "not", "now", "of", "off", "on", "once", "one", "only", "or",
    "other", "others", "our", "ours", "ourselves", "out", "over", "own", "part",
    "people", "put", "rather", "really", "right", "said", "same", "say", "see", "seen",
    "set", "shall", "shan", "she", "should", "shouldn", "show", "since", "so", "some",
    "still", "such", "take", "taken", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "thing", "things", "this", "those",
    "though", "through", "thus", "time", "to", "too", "two", "under", "until", "up",
    "upon", "us", "use", "used", "uses", "using", "very", "via", "want", "wants", "was",
    "wasn", "way", "ways", "we", "well", "were", "weren", "what", "when", "where",
    "whether", "which", "while", "who", "whom", "whose", "why", "will", "with", "within",
    "without", "won", "would", "wouldn", "yes", "yet", "you", "your", "yours", "yourself",
    "yourselves",
];

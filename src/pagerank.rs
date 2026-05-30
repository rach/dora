//! Aider-style PageRank over the file-level symbol graph.
//!
//! Built from the `links` table joined to `chunks` + `files`. Edge weight is a count of
//! resolved links between two files, multiplied by identifier-quality heuristics (real-looking
//! names get more weight than `len`/`new`/`push`). Personalization biases the random surfer
//! toward `focus_paths` (e.g. the file the agent is currently editing) — that's the bit that
//! makes `repo_map(focus_paths=[...])` useful instead of just showing the same global outline
//! every time.
//!
//! Output: `file_id -> score`. Callers (search ranker, repo_map) then turn that into a chunk
//! ordering. Scores are not comparable across databases (separately-computed graphs).

use anyhow::Result;
use rusqlite::Connection;
use std::collections::{HashMap, HashSet};

const ITERATIONS: usize = 30;
const DAMPING: f64 = 0.85;
const FOCUS_BOOST: f64 = 50.0;
const REAL_IDENT_BOOST: f64 = 10.0;
const PRIVATE_PENALTY: f64 = 0.1;
const GENERIC_THRESHOLD: usize = 5; // symbol defined in ≥N files → treated as generic
const GENERIC_PENALTY: f64 = 0.1;

/// Run PageRank and return a `file_id -> score` map. `focus_paths` are file path prefixes
/// the user wants ranked higher (matched via SQL `LIKE prefix%`). Pass an empty slice for an
/// unpersonalized run.
///
/// Always emits a row for every file_id, even with zero in-edges (uniform damping mass).
pub fn compute(conn: &Connection, focus_paths: &[String]) -> Result<HashMap<i64, f64>> {
    // file_id -> path (kept for personalization lookup).
    let file_paths: HashMap<i64, String> = {
        let mut stmt = conn.prepare("SELECT id, path FROM files")?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
        })?;
        let mut out = HashMap::new();
        for r in rows {
            let (id, path) = r?;
            out.insert(id, path);
        }
        out
    };

    if file_paths.is_empty() {
        return Ok(HashMap::new());
    }

    // How many distinct files define each symbol — used for the "generic name" penalty.
    let symbol_def_count: HashMap<String, usize> = {
        let mut stmt = conn.prepare(
            "SELECT symbol, COUNT(DISTINCT file_id) FROM chunks \
             WHERE symbol IS NOT NULL AND symbol != '' GROUP BY symbol",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)? as usize))
        })?;
        let mut out = HashMap::new();
        for r in rows {
            let (sym, n) = r?;
            out.insert(sym, n);
        }
        out
    };

    // Pull resolved edges with source_file, target_file, target_symbol.
    let raw_edges: Vec<(i64, i64, String)> = {
        let mut stmt = conn.prepare(
            "SELECT sc.file_id, tc.file_id, l.target_symbol \
             FROM links l \
             JOIN chunks sc ON sc.id = l.source_chunk_id \
             JOIN chunks tc ON tc.id = l.target_chunk_id \
             WHERE l.target_chunk_id IS NOT NULL",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, Option<String>>(2)?.unwrap_or_default(),
            ))
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        out
    };

    // Aggregate (src, dst) → weight with identifier heuristics applied.
    let mut weighted: HashMap<(i64, i64), f64> = HashMap::new();
    for (src, dst, sym) in &raw_edges {
        if src == dst {
            // Self-edges (within-file calls) add no ranking signal.
            continue;
        }
        let mut w = 1.0_f64;
        if is_real_identifier(sym) {
            w *= REAL_IDENT_BOOST;
        }
        if sym.starts_with('_') {
            w *= PRIVATE_PENALTY;
        }
        if symbol_def_count.get(sym).copied().unwrap_or(0) >= GENERIC_THRESHOLD {
            w *= GENERIC_PENALTY;
        }
        *weighted.entry((*src, *dst)).or_insert(0.0) += w;
    }

    // Build outbound adjacency. Each file's outbound weights are normalized to sum to 1 so
    // PageRank mass is conserved per node.
    let mut out_adj: HashMap<i64, Vec<(i64, f64)>> = HashMap::new();
    let mut out_sum: HashMap<i64, f64> = HashMap::new();
    for ((src, dst), w) in &weighted {
        out_adj.entry(*src).or_default().push((*dst, *w));
        *out_sum.entry(*src).or_insert(0.0) += w;
    }
    for (src, edges) in out_adj.iter_mut() {
        let s = out_sum.get(src).copied().unwrap_or(1.0);
        if s > 0.0 {
            for (_, w) in edges.iter_mut() {
                *w /= s;
            }
        }
    }

    // Personalization vector over EVERY file: focus_paths get FOCUS_BOOST, the rest 1.0.
    // Passing all files as seed entries preserves the "row per file" contract (isolated
    // files still appear) and lets the generic `compute_ppr` engine do the iteration.
    let focus_ids: HashSet<i64> = if focus_paths.is_empty() {
        HashSet::new()
    } else {
        file_paths
            .iter()
            .filter(|(_, p)| focus_paths.iter().any(|fp| p.starts_with(fp)))
            .map(|(id, _)| *id)
            .collect()
    };
    let seed: HashMap<i64, f64> = file_paths
        .keys()
        .map(|id| {
            let boost = if focus_ids.contains(id) { FOCUS_BOOST } else { 1.0 };
            (*id, boost)
        })
        .collect();

    let edges: Vec<(i64, i64, f64)> = weighted
        .into_iter()
        .map(|((s, d), w)| (s, d, w))
        .collect();

    Ok(compute_ppr(&edges, &seed, ITERATIONS, DAMPING))
}

/// Generic Personalized PageRank engine — the shared core behind both `repo_map` (file +
/// symbol graph, seeded by `focus_paths`) and the v0.10 graph-retrieval boost (chunk graph,
/// seeded by the search's top hits). Node ids are opaque `i64` (file ids or chunk ids).
///
/// `edges` are directed `(src, dst, weight)`; `seed` is the personalization vector (need not
/// be normalized — we normalize internally). Returns a score per node appearing in `edges`
/// or `seed`. Dangling mass redistributes uniformly (standard Brin/Page treatment).
pub fn compute_ppr(
    edges: &[(i64, i64, f64)],
    seed: &HashMap<i64, f64>,
    iterations: usize,
    damping: f64,
) -> HashMap<i64, f64> {
    let mut nodes: HashSet<i64> = HashSet::new();
    for (s, d, _) in edges {
        nodes.insert(*s);
        nodes.insert(*d);
    }
    for id in seed.keys() {
        nodes.insert(*id);
    }
    if nodes.is_empty() {
        return HashMap::new();
    }
    let n = nodes.len() as f64;

    // Out adjacency, normalized so each node's outgoing weights sum to 1.
    let mut out_adj: HashMap<i64, Vec<(i64, f64)>> = HashMap::new();
    let mut out_sum: HashMap<i64, f64> = HashMap::new();
    for (s, d, w) in edges {
        out_adj.entry(*s).or_default().push((*d, *w));
        *out_sum.entry(*s).or_insert(0.0) += w;
    }
    for (s, es) in out_adj.iter_mut() {
        let sum = out_sum.get(s).copied().unwrap_or(1.0);
        if sum > 0.0 {
            for (_, w) in es.iter_mut() {
                *w /= sum;
            }
        }
    }

    let seed_total: f64 = seed.values().sum();
    let pers = |id: i64| -> f64 {
        if seed_total > 0.0 {
            seed.get(&id).copied().unwrap_or(0.0) / seed_total
        } else {
            1.0 / n
        }
    };

    let teleport = 1.0 - damping;
    let mut rank: HashMap<i64, f64> = nodes.iter().map(|id| (*id, 1.0 / n)).collect();

    for _ in 0..iterations {
        let dangling_mass: f64 = rank
            .iter()
            .filter(|(id, _)| !out_adj.contains_key(*id))
            .map(|(_, r)| *r)
            .sum();
        let mut next: HashMap<i64, f64> = HashMap::with_capacity(rank.len());
        for &id in &nodes {
            let mut v = teleport * pers(id);
            v += damping * dangling_mass / n;
            next.insert(id, v);
        }
        for (src, es) in &out_adj {
            let src_rank = rank.get(src).copied().unwrap_or(0.0);
            for (dst, w) in es {
                if let Some(slot) = next.get_mut(dst) {
                    *slot += damping * src_rank * w;
                }
            }
        }
        rank = next;
    }

    rank
}

fn is_real_identifier(s: &str) -> bool {
    if s.len() < 8 {
        return false;
    }
    let has_under = s.contains('_');
    let mut transitions = 0;
    let mut prev_lower = false;
    for c in s.chars() {
        if c.is_ascii_uppercase() && prev_lower {
            transitions += 1;
        }
        prev_lower = c.is_ascii_lowercase();
    }
    has_under || transitions >= 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ppr_concentrates_mass_on_seed_neighborhood() {
        // Two disjoint triangles. Seed node 1 → mass should pool in {1,2,3}, not {4,5,6}.
        let edges = vec![
            (1, 2, 1.0), (2, 3, 1.0), (3, 1, 1.0),
            (4, 5, 1.0), (5, 6, 1.0), (6, 4, 1.0),
        ];
        let mut seed = HashMap::new();
        seed.insert(1i64, 1.0);
        let r = compute_ppr(&edges, &seed, 30, 0.85);
        let near: f64 = [1, 2, 3].iter().map(|i| r[i]).sum();
        let far: f64 = [4, 5, 6].iter().map(|i| r[i]).sum();
        assert!(near > far * 3.0, "seed cluster {near} should dominate far cluster {far}");
    }

    #[test]
    fn real_identifier_heuristics() {
        assert!(is_real_identifier("compute_pagerank"));
        assert!(is_real_identifier("ComputePagerank"));
        assert!(is_real_identifier("computeRank"));
        assert!(!is_real_identifier("foo"));
        assert!(!is_real_identifier("len"));
        assert!(!is_real_identifier("HashMap")); // < 8 chars
    }
}

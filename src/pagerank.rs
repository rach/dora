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

    // Personalization vector. `focus_paths` are matched as prefixes against file paths.
    let n = file_paths.len() as f64;
    let focus_ids: HashSet<i64> = if focus_paths.is_empty() {
        HashSet::new()
    } else {
        file_paths
            .iter()
            .filter(|(_, p)| focus_paths.iter().any(|fp| p.starts_with(fp)))
            .map(|(id, _)| *id)
            .collect()
    };
    let pers_total: f64 = if focus_ids.is_empty() {
        n
    } else {
        focus_ids.len() as f64 * FOCUS_BOOST + (n - focus_ids.len() as f64)
    };
    let personalization = |id: i64| -> f64 {
        let boost = if focus_ids.contains(&id) {
            FOCUS_BOOST
        } else {
            1.0
        };
        boost / pers_total
    };

    // Initialize uniform; iterate. Sinks (no outgoing edges) leak mass to the whole graph
    // via the standard "dangling redistribution" trick.
    let mut rank: HashMap<i64, f64> =
        file_paths.keys().map(|id| (*id, 1.0 / n)).collect();
    let damping = DAMPING;
    let teleport = 1.0 - damping;

    for _ in 0..ITERATIONS {
        let mut next: HashMap<i64, f64> = HashMap::with_capacity(rank.len());
        let dangling_mass: f64 = rank
            .iter()
            .filter(|(id, _)| !out_adj.contains_key(*id))
            .map(|(_, r)| *r)
            .sum();
        for &id in file_paths.keys() {
            // Base teleport (personalized).
            let mut v = teleport * personalization(id);
            // Dangling mass redistributed uniformly (could also be personalized — keeps it
            // simple and matches the standard Brin/Page treatment).
            v += damping * dangling_mass / n;
            next.insert(id, v);
        }
        // Add contribution from each source's outgoing distribution.
        for (src, edges) in &out_adj {
            let src_rank = rank.get(src).copied().unwrap_or(0.0);
            for (dst, w) in edges {
                if let Some(slot) = next.get_mut(dst) {
                    *slot += damping * src_rank * w;
                }
            }
        }
        rank = next;
    }

    Ok(rank)
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
    fn real_identifier_heuristics() {
        assert!(is_real_identifier("compute_pagerank"));
        assert!(is_real_identifier("ComputePagerank"));
        assert!(is_real_identifier("computeRank"));
        assert!(!is_real_identifier("foo"));
        assert!(!is_real_identifier("len"));
        assert!(!is_real_identifier("HashMap")); // < 8 chars
    }
}

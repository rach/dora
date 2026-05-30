//! Layer B — derived document-graph edges, recomputed wholesale per source at index time.
//!
//! Two edge sources, both dependency-free:
//!   * **keyphrase** — RAKE (Rapid Automatic Keyword Extraction): statistical, no model.
//!     Chunks sharing a keyphrase get an edge; weight = number of shared phrases.
//!   * **similarity** — kNN over the embeddings already stored in `chunks_vec`; weight = cosine.
//!
//! An optional **entity** edge source (GLiNER, ~300 MB ONNX) is gated behind
//! `[graph] entities = true` and is not implemented yet (stubbed with a warning).
//!
//! These complement the authored wikilink graph (Layer A). The Personalized-PageRank
//! retrieval boost (Layer C) consumes all edge kinds, weighting authored links highest.

use anyhow::Result;
use std::collections::{HashMap, HashSet};

use crate::search::STOPWORDS;
use crate::store::Store;

/// Top keyphrases extracted per chunk.
const KEYPHRASE_MAX_PER_CHUNK: usize = 8;
/// Minimum phrase length (chars) — drops 1-2 char noise.
const KEYPHRASE_MIN_LEN: usize = 4;
/// Phrases appearing in more than this many chunks are dropped as hubs (e.g. boilerplate).
const PHRASE_DF_CAP: usize = 50;
/// Minimum shared salient words for a keyphrase edge. 1 shared word is mostly noise
/// (measured: ~16% same-doc precision); ≥2 lifts precision sharply. Eval-tunable.
const KEYPHRASE_MIN_SHARED: f64 = 2.0;
/// kNN neighbours per chunk for similarity edges.
const SIM_N: usize = 8;
/// Minimum cosine for a similarity edge.
const SIM_THRESHOLD: f32 = 0.75;

/// Wipe and rebuild every derived edge for the source behind `store`. Returns the edge count.
/// `entities_enabled` reflects `[graph] entities` — currently warns (no model shipped).
pub fn rebuild_derived_edges(store: &Store, entities_enabled: bool) -> Result<usize> {
    store.clear_graph_edges()?;

    let chunks = store.all_chunks_for_graph()?;
    if chunks.is_empty() {
        return Ok(0);
    }

    let mut edges: Vec<(i64, i64, &'static str, f64)> = Vec::new();
    keyphrase_edges(&chunks, &mut edges);
    similarity_edges(store, &mut edges)?;

    if entities_enabled {
        eprintln!(
            "dora: [graph] entities = true requested, but the GLiNER entity extractor is not \
             implemented yet — skipping entity edges. (keyphrase + similarity edges still built)"
        );
    }

    let n = edges.len();
    store.insert_graph_edges(&edges)?;
    Ok(n)
}

// ---------------- keyphrase edges (RAKE) ----------------

/// Build keyphrase co-occurrence edges. RAKE selects each chunk's salient phrases; we connect
/// chunks at the *word* level (the distinct content words of those top phrases), because exact
/// multi-word phrases rarely repeat verbatim across documents ("investor round" vs "investor
/// round closes") while their salient words do. Weight = number of shared salient words.
/// Hub words (in > `PHRASE_DF_CAP` chunks) are dropped; undirected, deduped to `(min, max)`.
fn keyphrase_edges(chunks: &[(i64, String)], out: &mut Vec<(i64, i64, &'static str, f64)>) {
    // salient word -> chunk ids
    let mut inverted: HashMap<String, Vec<i64>> = HashMap::new();
    for (id, content) in chunks {
        let mut terms: HashSet<String> = HashSet::new();
        for phrase in keyphrases(content, KEYPHRASE_MAX_PER_CHUNK) {
            for w in phrase.split_whitespace() {
                if w.len() >= 3 {
                    terms.insert(w.to_string());
                }
            }
        }
        for t in terms {
            inverted.entry(t).or_default().push(*id);
        }
    }

    // pair -> shared salient-word count
    let mut pair_weight: HashMap<(i64, i64), f64> = HashMap::new();
    for (_term, ids) in &inverted {
        if ids.len() < 2 || ids.len() > PHRASE_DF_CAP {
            continue; // singletons add no edge; hub words add noise
        }
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                let (a, b) = order(ids[i], ids[j]);
                if a != b {
                    *pair_weight.entry((a, b)).or_insert(0.0) += 1.0;
                }
            }
        }
    }

    for ((a, b), w) in pair_weight {
        if w >= KEYPHRASE_MIN_SHARED {
            out.push((a, b, "keyphrase", w));
        }
    }
}

/// RAKE keyphrase extraction. Splits text into candidate phrases at stopwords and
/// punctuation, scores each word by `degree / frequency`, sums word scores per phrase, and
/// returns the top-`top_n` distinct phrases. Pure statistics — no model.
fn keyphrases(text: &str, top_n: usize) -> Vec<String> {
    let stop: HashSet<&&str> = STOPWORDS.iter().collect();

    // Candidate phrases: runs of content words, broken by stopwords / non-alphanumeric.
    let mut phrases: Vec<Vec<String>> = Vec::new();
    let mut current: Vec<String> = Vec::new();
    let mut flush = |cur: &mut Vec<String>, out: &mut Vec<Vec<String>>| {
        if !cur.is_empty() {
            out.push(std::mem::take(cur));
        }
    };
    for raw in text.split(|c: char| !c.is_alphanumeric()) {
        if raw.is_empty() {
            continue;
        }
        let w = raw.to_lowercase();
        if w.len() < 2 || stop.contains(&w.as_str()) || !w.chars().any(|c| c.is_alphabetic()) {
            flush(&mut current, &mut phrases);
        } else {
            current.push(w);
        }
    }
    flush(&mut current, &mut phrases);

    if phrases.is_empty() {
        return Vec::new();
    }

    // Word scores: freq = occurrences, degree = sum of containing-phrase lengths (RAKE).
    let mut freq: HashMap<String, usize> = HashMap::new();
    let mut degree: HashMap<String, usize> = HashMap::new();
    for p in &phrases {
        let plen = p.len();
        for w in p {
            *freq.entry(w.clone()).or_insert(0) += 1;
            *degree.entry(w.clone()).or_insert(0) += plen;
        }
    }

    let mut scored: Vec<(String, f64)> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for p in &phrases {
        let phrase = p.join(" ");
        if phrase.len() < KEYPHRASE_MIN_LEN || !seen.insert(phrase.clone()) {
            continue;
        }
        let score: f64 = p
            .iter()
            .map(|w| degree[w] as f64 / freq[w] as f64)
            .sum();
        scored.push((phrase, score));
    }
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    scored.into_iter().take(top_n).map(|(p, _)| p).collect()
}

// ---------------- similarity edges (kNN over stored vectors) ----------------

/// kNN similarity edges. For each chunk, fetch its top-`SIM_N` ANN neighbours and keep those
/// with cosine ≥ `SIM_THRESHOLD`. Undirected, deduped to `(min_id, max_id)` keeping the max
/// cosine seen. Reuses the embeddings already in `chunks_vec` — no recompute.
fn similarity_edges(store: &Store, out: &mut Vec<(i64, i64, &'static str, f64)>) -> Result<()> {
    let all = store.all_chunk_embeddings()?;
    if all.len() < 2 {
        return Ok(());
    }
    let by_id: HashMap<i64, &Vec<f32>> = all.iter().map(|(id, v)| (*id, v)).collect();

    let mut pair_cos: HashMap<(i64, i64), f64> = HashMap::new();
    for (id, vec) in &all {
        let neighbors = store.search_ann(vec, SIM_N + 1, None)?;
        for nb in neighbors {
            if nb == *id {
                continue;
            }
            let Some(nbvec) = by_id.get(&nb) else { continue };
            let cos = cosine(vec, nbvec);
            if cos < SIM_THRESHOLD {
                continue;
            }
            let (a, b) = order(*id, nb);
            let e = pair_cos.entry((a, b)).or_insert(0.0);
            *e = e.max(cos as f64);
        }
    }
    for ((a, b), cos) in pair_cos {
        out.push((a, b, "similarity", cos));
    }
    Ok(())
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn order(a: i64, b: i64) -> (i64, i64) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rake_surfaces_multiword_keyphrases() {
        let text = "Personalized PageRank spreads activation across the knowledge graph. \
                    The knowledge graph connects passages. Personalized PageRank is the core.";
        let phrases = keyphrases(text, 5);
        // Multi-word content phrases should rank above bare common words.
        assert!(
            phrases.iter().any(|p| p.contains("knowledge graph")),
            "expected 'knowledge graph' in {phrases:?}"
        );
        assert!(
            phrases.iter().any(|p| p.contains("personalized pagerank")),
            "expected 'personalized pagerank' in {phrases:?}"
        );
        // Stopwords must not appear as standalone phrases.
        assert!(!phrases.iter().any(|p| p == "the" || p == "across"));
    }

    #[test]
    fn keyphrase_edges_connect_chunks_sharing_a_phrase() {
        let chunks = vec![
            (1i64, "Series A fundraising memo for the investor round".to_string()),
            (2i64, "The investor round closes after fundraising diligence".to_string()),
            (3i64, "Unrelated note about kubernetes pod scheduling".to_string()),
        ];
        let mut edges = Vec::new();
        keyphrase_edges(&chunks, &mut edges);
        // Chunks 1 and 2 share fundraising/investor vocabulary → an edge; 3 is disjoint.
        assert!(
            edges.iter().any(|(a, b, k, _)| *a == 1 && *b == 2 && *k == "keyphrase"),
            "expected a keyphrase edge between 1 and 2: {edges:?}"
        );
        assert!(
            !edges.iter().any(|(a, b, _, _)| *a == 3 || *b == 3),
            "chunk 3 shares no phrase, should have no edges: {edges:?}"
        );
    }

    #[test]
    fn cosine_basic() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
    }
}

//! Adaptive markdown chunker (was `src/chunk.rs` through v0.1).
//!
//! Three behaviors, picked per-file:
//!   - Tiny files become one atomic chunk.
//!   - Files with headings split on heading boundaries; each section becomes one chunk.
//!   - Sections that still exceed `target_bytes` recursive-split on paragraph boundaries with overlap.
//!
//! Code fences (``` and ~~~) are treated as atomic — we never break inside one.
//!
//! Frontmatter-only "spec sheet" files (e.g. kepano-style movie/place notes) are detected and
//! their YAML is synthesized into natural-language prose so the embedder + FTS see real signal.

use regex::Regex;
use std::sync::OnceLock;

use crate::chunk::{Chunk, ChunkKind, Chunker, EdgeKind, EdgeSpec};
use crate::config::ChunkingConfig;

/// Holds the size knobs for chunking. One per `dora` run; configured from `ChunkingConfig`.
pub struct MarkdownChunker {
    pub target_bytes: usize,
    pub atomic_below_bytes: usize,
    pub overlap_bytes: usize,
}

impl MarkdownChunker {
    pub fn from_config(cfg: &ChunkingConfig) -> Self {
        Self {
            target_bytes: cfg.target_bytes,
            atomic_below_bytes: cfg.atomic_below_bytes,
            overlap_bytes: cfg.overlap_bytes,
        }
    }
}

impl Chunker for MarkdownChunker {
    fn chunk(&self, text: &str, rel_path: &str) -> Vec<Chunk> {
        if text.trim().is_empty() {
            return Vec::new();
        }

        // Synthesis title uses the basename without extension — strip here so callers can
        // pass the path-with-extension shape the trait expects.
        let rel_path_no_ext: String = std::path::Path::new(rel_path)
            .with_extension("")
            .to_string_lossy()
            .into_owned();

        if let (Some(fm), body) = split_frontmatter(text) {
            if body.trim().len() < 50 {
                let synthesized = synthesize_from_frontmatter(&rel_path_no_ext, fm);
                if !synthesized.trim().is_empty() {
                    return vec![prose_chunk(0, String::new(), synthesized, 0, text.len())];
                }
            }
        }

        if text.len() <= self.atomic_below_bytes {
            return vec![prose_chunk(
                0,
                String::new(),
                text.to_string(),
                0,
                text.len(),
            )];
        }

        let sections = split_by_headings(text);

        let mut out = Vec::new();
        let mut idx = 0usize;

        for section in sections {
            if section.content.len() <= self.target_bytes {
                out.push(prose_chunk(
                    idx,
                    section.heading_path.clone(),
                    section.content,
                    section.start_byte,
                    section.end_byte,
                ));
                idx += 1;
                continue;
            }

            for window in self.recursive_split(&section) {
                out.push(prose_chunk(
                    idx,
                    section.heading_path.clone(),
                    window.content,
                    window.start_byte,
                    window.end_byte,
                ));
                idx += 1;
            }
        }

        out
    }

    /// Wikilink + markdown-link edges. Parses `[[Note]]`, `[[folder/Note]]`,
    /// `[[Note#heading]]`, `[[Note|alias]]`, and `[text](note.md)` from the raw file,
    /// skipping anything inside a code fence, and attributes each to the chunk whose byte
    /// span contains the match. Targets resolve by note title/path in `Store::resolve_wikilinks`.
    fn edges(&self, text: &str, _rel_path: &str, chunks: &[Chunk]) -> Vec<EdgeSpec> {
        if chunks.is_empty() {
            return Vec::new();
        }
        let fenced = fenced_byte_ranges(text);
        let mut out = Vec::new();
        for (offset, target_symbol, target_path) in find_links(text) {
            if fenced.iter().any(|(s, e)| offset >= *s && offset < *e) {
                continue;
            }
            out.push(EdgeSpec {
                source_chunk_idx: chunk_for_offset(chunks, offset),
                kind: EdgeKind::Wikilink,
                target_symbol,
                target_path,
            });
        }
        out
    }
}

/// Byte ranges covered by fenced code blocks (``` / ~~~), so links inside them are ignored.
fn fenced_byte_ranges(text: &str) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut fence_start: Option<usize> = None;
    for (ls, le) in iter_line_spans(text) {
        let trimmed = text[ls..le].trim_end_matches('\n').trim_end_matches('\r');
        if is_fence_line(trimmed) {
            match fence_start {
                None => fence_start = Some(ls),
                Some(s) => {
                    ranges.push((s, le));
                    fence_start = None;
                }
            }
        }
    }
    if let Some(s) = fence_start {
        ranges.push((s, text.len())); // unterminated fence → treat rest of file as fenced
    }
    ranges
}

/// Index of the chunk whose `[start_byte, end_byte)` span contains `offset`. Falls back to the
/// last chunk starting at or before `offset` (links in heading lines sit in inter-chunk gaps),
/// and to chunk 0 if nothing precedes it.
fn chunk_for_offset(chunks: &[Chunk], offset: usize) -> usize {
    for c in chunks {
        if offset >= c.start_byte && offset < c.end_byte {
            return c.idx;
        }
    }
    chunks
        .iter()
        .filter(|c| c.start_byte <= offset)
        .max_by_key(|c| c.start_byte)
        .map(|c| c.idx)
        .unwrap_or(0)
}

/// Extract `(byte_offset, target_title, target_path)` for every wikilink + markdown-link.
/// `target_title` is the basename without extension (for title resolution); `target_path` is
/// `Some(..)` when the link carries path information (`[[folder/Note]]` or `[text](note.md)`).
fn find_links(text: &str) -> Vec<(usize, String, Option<String>)> {
    let mut out = Vec::new();
    for caps in wikilink_regex().captures_iter(text) {
        let m = caps.get(0).unwrap();
        let raw = caps.get(1).unwrap().as_str().trim();
        if raw.is_empty() {
            continue;
        }
        let (title, path) = split_target(raw);
        out.push((m.start(), title, path));
    }
    for caps in mdlink_regex().captures_iter(text) {
        let m = caps.get(0).unwrap();
        let raw = caps.get(1).unwrap().as_str().trim();
        if raw.is_empty() {
            continue;
        }
        let (title, _) = split_target(raw);
        out.push((m.start(), title, Some(raw.to_string())));
    }
    out
}

/// `"folder/Note"` → (`"Note"`, `Some("folder/Note")`); `"Note"` → (`"Note"`, `None`).
/// Strips a trailing `.md` from the title component.
fn split_target(raw: &str) -> (String, Option<String>) {
    let last = raw.rsplit('/').next().unwrap_or(raw);
    let title = last.strip_suffix(".md").unwrap_or(last).to_string();
    let path = if raw.contains('/') {
        Some(raw.to_string())
    } else {
        None
    };
    (title, path)
}

fn wikilink_regex() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    // [[target]] with optional #heading and |alias; target excludes ] | #
    R.get_or_init(|| Regex::new(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]").unwrap())
}

fn mdlink_regex() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    // [text](path.md) optionally with #fragment; capture the .md path
    R.get_or_init(|| Regex::new(r"\[[^\]]*\]\(([^)#]+\.md)(?:#[^)]*)?\)").unwrap())
}

fn prose_chunk(
    idx: usize,
    heading_path: String,
    content: String,
    start_byte: usize,
    end_byte: usize,
) -> Chunk {
    Chunk {
        idx,
        heading_path,
        content,
        start_byte,
        end_byte,
        kind: ChunkKind::Prose,
        symbol: None,
        parent_chunk_idx: None,
    }
}

// ------------- heading-aware section splitting -------------

struct Section {
    heading_path: String,
    content: String,
    start_byte: usize,
    end_byte: usize,
}

fn split_by_headings(text: &str) -> Vec<Section> {
    let heading_re = heading_regex();
    let mut sections = Vec::new();
    let mut heading_stack: Vec<(u8, String)> = Vec::new();
    let mut section_start: usize = 0;
    let mut section_heading_path = String::new();
    let mut in_fence = false;

    for (ls, le) in iter_line_spans(text) {
        let line = &text[ls..le];
        let trimmed = line.trim_end_matches('\n').trim_end_matches('\r');

        if is_fence_line(trimmed) {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            continue;
        }

        if let Some(caps) = heading_re.captures(trimmed) {
            // Close the previous section: [section_start, ls)
            if ls > section_start {
                let body = &text[section_start..ls];
                if !body.trim().is_empty() {
                    sections.push(Section {
                        heading_path: section_heading_path.clone(),
                        content: body.to_string(),
                        start_byte: section_start,
                        end_byte: ls,
                    });
                }
            }
            // Update heading path
            let level = caps.get(1).unwrap().as_str().len() as u8;
            let title = caps.get(2).unwrap().as_str().trim().to_string();
            while let Some((lvl, _)) = heading_stack.last() {
                if *lvl >= level {
                    heading_stack.pop();
                } else {
                    break;
                }
            }
            heading_stack.push((level, title));
            section_heading_path = heading_stack
                .iter()
                .map(|(_, t)| t.as_str())
                .collect::<Vec<_>>()
                .join(" > ");
            section_start = le; // section starts AFTER the heading line
        }
    }

    if section_start < text.len() {
        let body = &text[section_start..];
        if !body.trim().is_empty() {
            sections.push(Section {
                heading_path: section_heading_path,
                content: body.to_string(),
                start_byte: section_start,
                end_byte: text.len(),
            });
        }
    }

    if sections.is_empty() {
        // no headings or all-blank tail — treat whole file as one section
        sections.push(Section {
            heading_path: String::new(),
            content: text.to_string(),
            start_byte: 0,
            end_byte: text.len(),
        });
    }

    sections
}

// ------------- recursive paragraph split with overlap -------------

struct Window {
    content: String,
    start_byte: usize,
    end_byte: usize,
}

struct Block {
    content: String,
    start_byte: usize,
    end_byte: usize,
}

impl MarkdownChunker {
    fn recursive_split(&self, section: &Section) -> Vec<Window> {
        let blocks = paragraph_blocks(&section.content, section.start_byte);
        if blocks.is_empty() {
            return Vec::new();
        }

        let mut windows: Vec<Window> = Vec::new();
        let mut cur: Option<Window> = None;

        for block in blocks {
            let cur_len = cur.as_ref().map(|w| w.content.len()).unwrap_or(0);
            let separator_cost = if cur_len > 0 { 2 } else { 0 };
            let would_be = cur_len + separator_cost + block.content.len();

            if cur_len > 0 && would_be > self.target_bytes {
                let prev = cur.take().unwrap();
                let prev_content_snapshot = prev.content.clone();
                windows.push(prev);

                // Start next window with overlap carried from end of previous, aligned to paragraph
                let mut next = Window {
                    content: String::new(),
                    start_byte: block.start_byte,
                    end_byte: block.start_byte,
                };
                if self.overlap_bytes > 0 && prev_content_snapshot.len() > self.overlap_bytes {
                    let raw = prev_content_snapshot.len() - self.overlap_bytes;
                    // Walk forward to the next char boundary so the slice is valid UTF-8.
                    let mut cut_from = raw;
                    while cut_from < prev_content_snapshot.len()
                        && !prev_content_snapshot.is_char_boundary(cut_from)
                    {
                        cut_from += 1;
                    }
                    let aligned = prev_content_snapshot[cut_from..]
                        .find("\n\n")
                        .map(|p| cut_from + p + 2)
                        .unwrap_or(prev_content_snapshot.len());
                    if aligned < prev_content_snapshot.len() {
                        next.content.push_str(&prev_content_snapshot[aligned..]);
                    }
                }
                cur = Some(next);
            }

            let cur_ref = cur.get_or_insert(Window {
                content: String::new(),
                start_byte: block.start_byte,
                end_byte: block.start_byte,
            });
            if !cur_ref.content.is_empty() {
                cur_ref.content.push_str("\n\n");
            } else {
                cur_ref.start_byte = block.start_byte;
            }
            cur_ref.content.push_str(&block.content);
            cur_ref.end_byte = block.end_byte;
        }

        if let Some(w) = cur {
            if !w.content.trim().is_empty() {
                windows.push(w);
            }
        }

        windows
    }
}

fn paragraph_blocks(text: &str, base_offset: usize) -> Vec<Block> {
    let mut blocks: Vec<Block> = Vec::new();
    let mut buf = String::new();
    let mut buf_start: Option<usize> = None;
    let mut buf_end: usize = 0;
    let mut in_fence = false;

    let flush = |buf: &mut String, start: &mut Option<usize>, end: usize, out: &mut Vec<Block>| {
        if !buf.trim().is_empty() {
            out.push(Block {
                content: buf.trim_end_matches('\n').to_string(),
                start_byte: start.unwrap_or(0),
                end_byte: end,
            });
        }
        buf.clear();
        *start = None;
    };

    for (ls, le) in iter_line_spans(text) {
        let line = &text[ls..le];
        let trimmed = line.trim_end_matches('\n').trim_end_matches('\r');
        let is_blank = trimmed.trim().is_empty();
        let fence = is_fence_line(trimmed);

        if fence {
            if buf_start.is_none() {
                buf_start = Some(base_offset + ls);
            }
            buf.push_str(line);
            buf_end = base_offset + le;
            in_fence = !in_fence;
            continue;
        }

        if in_fence {
            buf.push_str(line);
            buf_end = base_offset + le;
            continue;
        }

        if is_blank {
            if !buf.is_empty() {
                flush(&mut buf, &mut buf_start, buf_end, &mut blocks);
            }
        } else {
            if buf_start.is_none() {
                buf_start = Some(base_offset + ls);
            }
            buf.push_str(line);
            buf_end = base_offset + le;
        }
    }

    if !buf.is_empty() {
        flush(&mut buf, &mut buf_start, buf_end, &mut blocks);
    }

    blocks
}

// ------------- shared helpers -------------

fn iter_line_spans(text: &str) -> Vec<(usize, usize)> {
    let mut starts = vec![0usize];
    for (i, b) in text.bytes().enumerate() {
        if b == b'\n' && i + 1 < text.len() {
            starts.push(i + 1);
        }
    }
    starts.push(text.len());
    starts.windows(2).map(|w| (w[0], w[1])).collect()
}

fn heading_regex() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"^(#{1,6})\s+(.+?)\s*#*\s*$").unwrap())
}

fn is_fence_line(line: &str) -> bool {
    let t = line.trim_start();
    t.starts_with("```") || t.starts_with("~~~")
}

// ------------- frontmatter handling for body-trivial files -------------

fn split_frontmatter(text: &str) -> (Option<&str>, &str) {
    let spans = iter_line_spans(text);
    if spans.is_empty() {
        return (None, text);
    }
    let (ls0, le0) = spans[0];
    let first = text[ls0..le0]
        .trim_end_matches('\n')
        .trim_end_matches('\r')
        .trim();
    if first != "---" {
        return (None, text);
    }
    for &(ls, le) in spans.iter().skip(1) {
        let line = text[ls..le]
            .trim_end_matches('\n')
            .trim_end_matches('\r')
            .trim();
        if line == "---" {
            let fm = &text[le0..ls];
            let body = &text[le.min(text.len())..];
            return (Some(fm), body);
        }
    }
    (None, text)
}

fn synthesize_from_frontmatter(rel_path_no_ext: &str, fm: &str) -> String {
    let title_raw = std::path::Path::new(rel_path_no_ext)
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| rel_path_no_ext.to_string());
    let title = title_raw.replace(['_', '-'], " ");

    let mut fields: Vec<(String, Vec<String>)> = Vec::new();

    for raw_line in fm.lines() {
        let line = raw_line.trim_end_matches('\r');
        if line.trim().is_empty() {
            continue;
        }
        let trimmed = line.trim_start();

        // List item under the previous key.
        if let Some(rest) = trimmed.strip_prefix("- ") {
            let v = clean_yaml_value(rest);
            if !v.is_empty() {
                if let Some((_, values)) = fields.last_mut() {
                    values.push(v);
                }
            }
            continue;
        }
        // key: value (value may be empty if a list follows)
        if let Some(colon) = trimmed.find(':') {
            let key = trimmed[..colon].trim().to_string();
            let rest = trimmed[colon + 1..].trim();
            let mut values = Vec::new();
            if !rest.is_empty() {
                let v = clean_yaml_value(rest);
                if !v.is_empty() {
                    values.push(v);
                }
            }
            fields.push((key, values));
        }
    }

    let mut parts = vec![title];
    for (key, values) in fields {
        if values.is_empty() {
            continue;
        }
        // Skip pure-metadata keys that add only noise to the embedding.
        if NOISY_KEYS.contains(&key.as_str()) {
            continue;
        }
        let pretty_key = key.replace('_', " ");
        parts.push(format!(
            "{}: {}",
            capitalize_first(&pretty_key),
            values.join(", ")
        ));
    }
    parts.join(". ")
}

const NOISY_KEYS: &[&str] = &[
    "imdbId",
    "imdb_id",
    "url",
    "cover",
    "image",
    "coordinates",
    "created",
    "modified",
    "published",
    "last",
    "id",
    "uuid",
];

fn clean_yaml_value(s: &str) -> String {
    let s = s.trim();
    let s = s.trim_matches('"').trim_matches('\'');
    let s = s.trim_start_matches("[[").trim_end_matches("]]");
    s.trim().to_string()
}

fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => c.to_uppercase().chain(chars).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ChunkingConfig;

    fn chunker() -> MarkdownChunker {
        MarkdownChunker::from_config(&ChunkingConfig {
            target_bytes: 1500,
            atomic_below_bytes: 1600,
            overlap_bytes: 270,
        })
    }

    #[test]
    fn wikilinks_parsed_with_alias_heading_and_path() {
        let text = "# Note\n\nSee [[Other Note]] and [[folder/Deep Note]] and \
                    [[Aliased|shown text]] and [[Sectioned#heading]].\n\nAlso [a link](target.md).";
        let chunks = chunker().chunk(text, "note.md");
        let edges = chunker().edges(text, "note.md", &chunks);
        let targets: Vec<&str> = edges.iter().map(|e| e.target_symbol.as_str()).collect();
        assert!(
            targets.contains(&"Other Note"),
            "bare wikilink: {targets:?}"
        );
        assert!(
            targets.contains(&"Deep Note"),
            "path wikilink basename: {targets:?}"
        );
        assert!(targets.contains(&"Aliased"), "alias stripped: {targets:?}");
        assert!(
            targets.contains(&"Sectioned"),
            "heading stripped: {targets:?}"
        );
        assert!(targets.contains(&"target"), "md-link basename: {targets:?}");
        // folder/Deep Note carries a path; bare Other Note does not.
        let deep = edges
            .iter()
            .find(|e| e.target_symbol == "Deep Note")
            .unwrap();
        assert_eq!(deep.target_path.as_deref(), Some("folder/Deep Note"));
        let other = edges
            .iter()
            .find(|e| e.target_symbol == "Other Note")
            .unwrap();
        assert_eq!(other.target_path, None);
        assert!(edges.iter().all(|e| e.kind == EdgeKind::Wikilink));
    }

    #[test]
    fn wikilinks_inside_code_fences_ignored() {
        let text = "# Note\n\nReal [[Linked]].\n\n```\nnot a [[FencedLink]] here\n```\n\nEnd.";
        let chunks = chunker().chunk(text, "n.md");
        let edges = chunker().edges(text, "n.md", &chunks);
        let targets: Vec<&str> = edges.iter().map(|e| e.target_symbol.as_str()).collect();
        assert!(targets.contains(&"Linked"));
        assert!(
            !targets.contains(&"FencedLink"),
            "fenced link must be skipped: {targets:?}"
        );
    }

    #[test]
    fn no_links_no_edges() {
        let text = "# Plain\n\nNothing to link here.";
        let chunks = chunker().chunk(text, "p.md");
        assert!(chunker().edges(text, "p.md", &chunks).is_empty());
    }
}

//! Code chunker — tree-sitter + tree-sitter-tags across six languages (Rust, Python, TS/JS,
//! Go, Java, Ruby).
//!
//! Each grammar ships a `queries/tags.scm` file that captures `@definition.*` and
//! `@reference.*` patterns. We lean on those instead of writing per-language queries ourselves
//! — the same approach Aider/Continue/Tabby use.
//!
//! Chunks are emitted from `@definition.*` tags (one chunk per function/method/class/etc.).
//! Edges are emitted from `@reference.*` tags (resolved against the `links` table in pass 2,
//! sub-slice C). Files with zero tag matches get a single `Module` chunk covering the whole
//! file so they're still searchable.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use tree_sitter::Language;
use tree_sitter_tags::{TagsConfiguration, TagsContext};

use super::{Chunk, ChunkKind, Chunker, EdgeKind, EdgeSpec};
use crate::config::ChunkingConfig;

// ---------- language registry ----------

/// Everything we need to chunk one language. `language` + `tags_query` come from the grammar
/// crate; `name` + `extensions` describe how dora picks this entry by file path.
struct LanguageSpec {
    name: &'static str,
    extensions: &'static [&'static str],
    language: Language,
    tags_query: &'static str,
    locals_query: &'static str,
}

fn registry() -> Vec<LanguageSpec> {
    vec![
        LanguageSpec {
            name: "rust",
            extensions: &["rs"],
            language: tree_sitter_rust::LANGUAGE.into(),
            tags_query: tree_sitter_rust::TAGS_QUERY,
            locals_query: "",
        },
        LanguageSpec {
            name: "python",
            extensions: &["py", "pyi"],
            language: tree_sitter_python::LANGUAGE.into(),
            tags_query: tree_sitter_python::TAGS_QUERY,
            locals_query: "",
        },
        LanguageSpec {
            name: "typescript",
            extensions: &["ts", "tsx"],
            language: tree_sitter_typescript::LANGUAGE_TSX.into(),
            tags_query: tree_sitter_typescript::TAGS_QUERY,
            locals_query: tree_sitter_typescript::LOCALS_QUERY,
        },
        LanguageSpec {
            name: "javascript",
            extensions: &["js", "jsx", "mjs", "cjs"],
            language: tree_sitter_javascript::LANGUAGE.into(),
            tags_query: tree_sitter_javascript::TAGS_QUERY,
            locals_query: tree_sitter_javascript::LOCALS_QUERY,
        },
        LanguageSpec {
            name: "go",
            extensions: &["go"],
            language: tree_sitter_go::LANGUAGE.into(),
            tags_query: tree_sitter_go::TAGS_QUERY,
            locals_query: "",
        },
        LanguageSpec {
            name: "java",
            extensions: &["java"],
            language: tree_sitter_java::LANGUAGE.into(),
            tags_query: tree_sitter_java::TAGS_QUERY,
            locals_query: "",
        },
        LanguageSpec {
            name: "ruby",
            extensions: &["rb"],
            language: tree_sitter_ruby::LANGUAGE.into(),
            tags_query: tree_sitter_ruby::TAGS_QUERY,
            locals_query: tree_sitter_ruby::LOCALS_QUERY,
        },
    ]
}

/// All code-source extensions across the registry. Mirrors `Mode::Code` extensions —
/// re-exposed here so `vault::list_entries` can use the registry as the source of truth.
pub fn code_extensions() -> Vec<&'static str> {
    registry()
        .iter()
        .flat_map(|s| s.extensions.iter().copied())
        .collect()
}

// ---------- chunker ----------

pub struct CodeChunker {
    /// One TagsConfiguration per language. Built once, reused across files.
    configs: Vec<LangEntry>,
    /// TagsContext is `!Sync` (holds a Parser). Mutex makes the whole chunker `Sync` so it
    /// can live in the MCP server's `Arc<>` state. Indexing is single-threaded today; if/when
    /// we parallelize, swap to a thread-local or pool.
    context: Mutex<TagsContext>,
}

struct LangEntry {
    name: &'static str,
    extensions: &'static [&'static str],
    config: TagsConfiguration,
}

impl CodeChunker {
    pub fn new(_cfg: &ChunkingConfig) -> Self {
        let mut configs = Vec::new();
        for spec in registry() {
            match TagsConfiguration::new(spec.language, spec.tags_query, spec.locals_query) {
                Ok(config) => configs.push(LangEntry {
                    name: spec.name,
                    extensions: spec.extensions,
                    config,
                }),
                Err(e) => {
                    eprintln!(
                        "warning: tree-sitter tags query for {} failed to compile: {e}",
                        spec.name
                    );
                }
            }
        }
        Self {
            configs,
            context: Mutex::new(TagsContext::new()),
        }
    }

    /// Pick the LangEntry whose extensions include the file's extension. `rel_path_no_ext`
    /// arrives without the dot, so we infer extension from a sibling helper or — easier —
    /// dispatch in `chunk()` where the caller still has the path.
    fn entry_for(&self, ext: &str) -> Option<&LangEntry> {
        self.configs
            .iter()
            .find(|e| e.extensions.iter().any(|x| *x == ext))
    }

    fn parse_chunks_and_edges(
        &self,
        text: &str,
        rel_path_no_ext: &str,
        ext: &str,
    ) -> (Vec<Chunk>, Vec<EdgeSpec>) {
        let Some(entry) = self.entry_for(ext) else {
            return (fallback_single_chunk(text, rel_path_no_ext), Vec::new());
        };

        let source = text.as_bytes();
        let mut ctx = self.context.lock().expect("tags context poisoned");
        let mut raw_tags: Vec<RawTag> = Vec::new();
        match ctx.generate_tags(&entry.config, source, None) {
            Ok((iter, _had_error)) => {
                for tag_result in iter {
                    let Ok(tag) = tag_result else { continue };
                    let kind_name = entry.config.syntax_type_name(tag.syntax_type_id);
                    let name = std::str::from_utf8(&source[tag.name_range.clone()])
                        .unwrap_or("")
                        .to_string();
                    raw_tags.push(RawTag {
                        is_definition: tag.is_definition,
                        kind_name: kind_name.to_string(),
                        name,
                        range_start: tag.range.start,
                        range_end: tag.range.end,
                    });
                }
            }
            Err(e) => {
                eprintln!("warning: tree-sitter failed on {rel_path_no_ext}.{ext}: {e}");
                return (fallback_single_chunk(text, rel_path_no_ext), Vec::new());
            }
        }
        drop(ctx);

        if raw_tags.is_empty() {
            return (fallback_single_chunk(text, rel_path_no_ext), Vec::new());
        }

        build_chunks_and_edges(text, rel_path_no_ext, raw_tags, entry.name)
    }
}

impl Chunker for CodeChunker {
    fn chunk(&self, text: &str, rel_path_no_ext: &str) -> Vec<Chunk> {
        let ext = guess_ext(rel_path_no_ext);
        self.parse_chunks_and_edges(text, rel_path_no_ext, &ext).0
    }

    fn edges(&self, text: &str, rel_path_no_ext: &str, _chunks: &[Chunk]) -> Vec<EdgeSpec> {
        let ext = guess_ext(rel_path_no_ext);
        self.parse_chunks_and_edges(text, rel_path_no_ext, &ext).1
    }
}

// ---------- chunk + edge assembly ----------

#[derive(Debug)]
struct RawTag {
    is_definition: bool,
    kind_name: String,
    name: String,
    range_start: usize,
    range_end: usize,
}

fn build_chunks_and_edges(
    text: &str,
    rel_path_no_ext: &str,
    mut tags: Vec<RawTag>,
    language: &str,
) -> (Vec<Chunk>, Vec<EdgeSpec>) {
    // Sort defs first so parent containment can be computed by walking left-to-right with a
    // small enclosing-stack. Ties: longer ranges first (outer scope precedes inner scope).
    tags.sort_by(|a, b| {
        a.range_start
            .cmp(&b.range_start)
            .then_with(|| b.range_end.cmp(&a.range_end))
    });

    // Split into definitions (become chunks) and references (become edges).
    let defs: Vec<&RawTag> = tags.iter().filter(|t| t.is_definition).collect();
    let refs: Vec<&RawTag> = tags.iter().filter(|t| !t.is_definition).collect();

    let sep = heading_separator(language);
    let mut chunks: Vec<Chunk> = Vec::with_capacity(defs.len());
    // Stack of (def_idx_in_chunks, range_end) for containment walking.
    let mut stack: Vec<(usize, usize)> = Vec::new();

    for def in &defs {
        // Pop scopes we've exited.
        while let Some(&(_, end)) = stack.last() {
            if end <= def.range_start {
                stack.pop();
            } else {
                break;
            }
        }
        let parent_idx = stack.last().map(|&(idx, _)| idx);
        // heading_path is the qualified prefix (parent's full name), NOT including self.
        // Renderers join `heading_path + sep + symbol` to get the full qualified identifier.
        let heading_path = if let Some(pidx) = parent_idx {
            let parent = &chunks[pidx];
            let parent_sym = parent.symbol.as_deref().unwrap_or("");
            if parent.heading_path.is_empty() {
                parent_sym.to_string()
            } else {
                format!("{}{}{}", parent.heading_path, sep, parent_sym)
            }
        } else {
            String::new()
        };
        // Clamp range to text byte length defensively (tree-sitter ranges are reliable but
        // we'd rather emit a degraded chunk than panic on a UTF-8 slice).
        let start = def.range_start.min(text.len());
        let end = def.range_end.min(text.len());
        let content = safe_slice(text, start, end).to_string();

        let chunk_idx = chunks.len();
        chunks.push(Chunk {
            idx: chunk_idx,
            heading_path,
            content,
            start_byte: start,
            end_byte: end,
            kind: map_def_kind(&def.kind_name, language),
            symbol: Some(def.name.clone()),
            parent_chunk_idx: parent_idx,
        });
        stack.push((chunk_idx, def.range_end));
    }

    // Edges: each reference's source_chunk_idx = innermost enclosing def chunk. References
    // outside any def (top-level imports etc.) get attached to chunk 0 if it exists — we
    // record them as file-level signal rather than dropping them.
    let mut edges: Vec<EdgeSpec> = Vec::with_capacity(refs.len());
    let mut def_stack: Vec<(usize, usize, usize)> = Vec::new(); // (chunk_idx, def_start, def_end)
    for (i, def) in defs.iter().enumerate() {
        def_stack.push((i, def.range_start, def.range_end));
    }
    // For O(log n) enclosing lookup we'd sort + binary search; defs.len() is small per file
    // (~hundreds), so a linear walk is fine and clearer.
    for r in &refs {
        let owner_idx = innermost_enclosing(&defs, r.range_start)
            .or(if chunks.is_empty() { None } else { Some(0) });
        let Some(idx) = owner_idx else { continue };
        edges.push(EdgeSpec {
            source_chunk_idx: idx,
            kind: map_ref_kind(&r.kind_name),
            target_symbol: r.name.clone(),
            target_path: None,
        });
    }

    // Always emit a top-of-file Module chunk if the first def isn't already at byte 0 and
    // there's leading content — covers Python/JS imports + top-level module docs.
    if let Some(first) = chunks.first() {
        if first.start_byte > 16 {
            let header = safe_slice(text, 0, first.start_byte).to_string();
            if !header.trim().is_empty() {
                // Insert a module chunk at the front; renumber later.
                let module_chunk = Chunk {
                    idx: 0,
                    heading_path: String::new(),
                    content: header,
                    start_byte: 0,
                    end_byte: first.start_byte,
                    kind: ChunkKind::Module,
                    symbol: Some(file_basename(rel_path_no_ext)),
                    parent_chunk_idx: None,
                };
                chunks.insert(0, module_chunk);
                // Bump indices + parent pointers + edge source_chunk_idx by 1.
                for (i, c) in chunks.iter_mut().enumerate() {
                    c.idx = i;
                    if let Some(p) = c.parent_chunk_idx.as_mut() {
                        *p += 1;
                    }
                }
                for e in edges.iter_mut() {
                    e.source_chunk_idx += 1;
                }
            }
        }
    }

    (chunks, edges)
}

fn innermost_enclosing(defs: &[&RawTag], pos: usize) -> Option<usize> {
    // defs are sorted outer-first; walk forward and remember the *last* one whose range
    // contains `pos`.
    let mut best: Option<usize> = None;
    for (i, d) in defs.iter().enumerate() {
        if d.range_start <= pos && pos < d.range_end {
            best = Some(i);
        } else if d.range_start > pos {
            break;
        }
    }
    best
}

fn fallback_single_chunk(text: &str, rel_path_no_ext: &str) -> Vec<Chunk> {
    if text.is_empty() {
        return Vec::new();
    }
    vec![Chunk {
        idx: 0,
        heading_path: String::new(),
        content: text.to_string(),
        start_byte: 0,
        end_byte: text.len(),
        kind: ChunkKind::Module,
        symbol: Some(file_basename(rel_path_no_ext)),
        parent_chunk_idx: None,
    }]
}

fn heading_separator(language: &str) -> &'static str {
    match language {
        "rust" | "ruby" => "::",
        "python" => ".",
        "java" => ".",
        _ => ".",
    }
}

fn map_def_kind(name: &str, _language: &str) -> ChunkKind {
    // Tag names come from the language's tags.scm. Across the 5 grammars we use, the
    // observed definition kinds are: function, method, class, interface, module, macro,
    // constant, type. Anything unknown falls back to Module.
    match name {
        "function" => ChunkKind::Function,
        "method" => ChunkKind::Method,
        "class" => ChunkKind::Class,
        "struct" => ChunkKind::Struct,
        "interface" => ChunkKind::Interface,
        "trait" => ChunkKind::Trait,
        "impl" => ChunkKind::Impl,
        "enum" => ChunkKind::Enum,
        "module" => ChunkKind::Module,
        "constant" => ChunkKind::Const,
        "macro" => ChunkKind::Macro,
        // Go uses `definition.type` for type aliases / structs; classify as Class so it
        // shows up in interface-style queries.
        "type" => ChunkKind::Class,
        _ => ChunkKind::Module,
    }
}

fn map_ref_kind(name: &str) -> EdgeKind {
    match name {
        "call" => EdgeKind::Calls,
        "implementation" => EdgeKind::Implements,
        "class" | "type" | "interface" => EdgeKind::References,
        "import" => EdgeKind::Imports,
        _ => EdgeKind::References,
    }
}

fn file_basename(rel_path: &str) -> String {
    let base = rel_path.rsplit(['/', '\\']).next().unwrap_or(rel_path);
    base.rsplit_once('.').map(|(stem, _)| stem).unwrap_or(base).to_string()
}

fn guess_ext(rel_path_no_ext: &str) -> String {
    // rel_path_no_ext has the extension already stripped by the caller. We need to recover
    // it. The vault layer stores stripped paths but we also stash the original ext via a
    // sidechannel? — simpler: dora's vault uses `path.with_extension("")` for the rel path
    // and stores `ext` separately. The chunker, however, only sees `rel_path_no_ext`.
    //
    // To keep the trait shape compatible with markdown's existing contract, we instead pass
    // the extension by appending it to the path before chunking — but markdown ignores
    // extensions. For the code chunker, the caller (chunk_file in main.rs) passes the path
    // WITH extension when source mode is `code`. See the `cmd_index` plumbing.
    rel_path_no_ext
        .rsplit_once('.')
        .map(|(_, ext)| ext.to_string())
        .unwrap_or_default()
}

fn safe_slice(text: &str, start: usize, end: usize) -> &str {
    // Walk forward/backward to char boundaries if tree-sitter handed us a mid-codepoint
    // offset (rare with UTF-8 source but defensive).
    let mut s = start;
    while s < text.len() && !text.is_char_boundary(s) {
        s += 1;
    }
    let mut e = end;
    while e > s && !text.is_char_boundary(e) {
        e -= 1;
    }
    &text[s..e]
}

// ---------- tests ----------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_smoke() {
        let src = "fn foo() {}\nfn bar() { foo(); }\n";
        let chunker = CodeChunker::new(&ChunkingConfig {
            target_bytes: 1800,
            atomic_below_bytes: 1600,
            overlap_bytes: 270,
        });
        let chunks = chunker.chunk(src, "x.rs");
        assert!(chunks.iter().any(|c| c.symbol.as_deref() == Some("foo")));
        assert!(chunks.iter().any(|c| c.symbol.as_deref() == Some("bar")));
        let edges = chunker.edges(src, "x.rs", &chunks);
        assert!(edges.iter().any(|e| e.target_symbol == "foo"));
    }

    #[test]
    fn ruby_smoke() {
        let src = "module Greeter\n  class Hello\n    def say(name)\n      puts \"hi #{name}\"\n    end\n  end\nend\n";
        let chunker = CodeChunker::new(&ChunkingConfig {
            target_bytes: 1800,
            atomic_below_bytes: 1600,
            overlap_bytes: 270,
        });
        let chunks = chunker.chunk(src, "g.rb");
        assert!(chunks.iter().any(|c| c.symbol.as_deref() == Some("Greeter")));
        assert!(chunks.iter().any(|c| c.symbol.as_deref() == Some("Hello")));
        assert!(chunks.iter().any(|c| c.symbol.as_deref() == Some("say")));
    }

    #[test]
    fn python_smoke() {
        let src = "def foo():\n    return 1\n\nclass Bar:\n    def baz(self):\n        foo()\n";
        let chunker = CodeChunker::new(&ChunkingConfig {
            target_bytes: 1800,
            atomic_below_bytes: 1600,
            overlap_bytes: 270,
        });
        let chunks = chunker.chunk(src, "m.py");
        assert!(chunks.iter().any(|c| c.symbol.as_deref() == Some("foo")));
        assert!(chunks.iter().any(|c| c.symbol.as_deref() == Some("Bar")));
        assert!(chunks.iter().any(|c| c.symbol.as_deref() == Some("baz")));
    }
}

//! Chunking layer. One trait, multiple impls dispatched by source [`Mode`].
//!
//! v0–v0.1 had a single concrete markdown chunker. v0.2 extracts the trait because a real
//! second impl (code, via tree-sitter) finally exists to motivate the shape — same discipline
//! we held for `Embedder`. Markdown sources continue to use [`markdown::MarkdownChunker`];
//! code sources will use [`code::CodeChunker`] (sub-slice B).

pub mod claude_code;
pub mod code;
pub mod codex;
pub mod markdown;

use std::path::Path;

use crate::config::Config;
use crate::mode::Mode;

/// Output of any chunker — a slice of source text plus enough metadata to render it back to
/// the user, embed it, store it, and (for code) wire it into the structural [`EdgeSpec`] graph.
#[derive(Debug, Clone)]
pub struct Chunk {
    /// 0-based index within the file's chunk list. Stable across re-indexes for unchanged files.
    pub idx: usize,
    /// "Projects > dora > Indexing" for markdown, or "crate::module::method" for code. Empty
    /// when the chunk lives at the top of the file before any structural anchor.
    pub heading_path: String,
    /// Raw chunk text. What gets stored, what we render as a snippet.
    pub content: String,
    /// Byte offset into the original file (for line-number rendering at search time).
    pub start_byte: usize,
    pub end_byte: usize,
    /// Semantic kind. `Prose` for markdown chunks today. Code chunkers (sub-slice B) populate
    /// with `Function`/`Method`/`Struct`/etc. Defaulting to `Prose` keeps the markdown path
    /// behaviorally unchanged through this refactor.
    pub kind: ChunkKind,
    /// Symbol name (function/struct/trait name) for code chunks. `None` for prose.
    pub symbol: Option<String>,
    /// In-memory index of the parent chunk in the same file (e.g., method inside class). Used
    /// during indexing to populate the DB's `parent_chunk_id` foreign key after both rows exist.
    pub parent_chunk_idx: Option<usize>,
}

/// What kind of source unit this chunk represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkKind {
    /// Free-form prose (markdown sections, frontmatter-synthesized content).
    Prose,
    Function,
    Method,
    Class,
    Struct,
    Trait,
    Interface,
    Impl,
    Enum,
    Module,
    Const,
    Macro,
}

/// Pre-resolution edge description emitted by code chunkers. Pass-1 ingest accepts these from
/// the chunker; pass-2 resolves `target_symbol`/`target_path` to a concrete `target_chunk_id`
/// in the DB. Markdown returns empty.
#[derive(Debug, Clone)]
pub struct EdgeSpec {
    /// Index (within the same file's chunk list) of the chunk that owns this edge.
    pub source_chunk_idx: usize,
    pub kind: EdgeKind,
    /// The textually-captured target — may be qualified (`foo::bar::Baz`) or unqualified (`Baz`).
    pub target_symbol: String,
    /// If the chunker could resolve the target's module/file from imports or fully-qualified
    /// names, this carries it. `None` means "unqualified — resolver does a name-only match."
    pub target_path: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeKind {
    Calls,
    References,
    Implements,
    Imports,
    /// Authored prose link: Obsidian `[[wikilink]]` or markdown `[text](note.md)`.
    /// Resolved by note title/path (not symbol) and kept distinct from code edges so
    /// PageRank's identifier heuristics never touch it.
    Wikilink,
}

/// All chunkers implement this. Sync because indexing is sync; `Send` because main spawns
/// workers later. Stateless after construction — config bound at `from_config` time.
///
/// `rel_path` is the file's path relative to the source root, with its extension intact.
/// Code chunkers need the extension to dispatch by language; markdown derives its synthesis
/// title from the basename. Embedding signal still uses the no-ext form (see
/// [`embedded_text`]) so the v0.1 hash contract is preserved for markdown sources.
pub trait Chunker: Send + Sync {
    fn chunk(&self, text: &str, rel_path: &str) -> Vec<Chunk>;

    /// Edge specs the chunker found in this file. Default returns empty — markdown overrides
    /// with no-op; code chunkers (sub-slice B) populate with calls/references/etc.
    fn edges(&self, _text: &str, _rel_path: &str, _chunks: &[Chunk]) -> Vec<EdgeSpec> {
        Vec::new()
    }
}

// Lets `&Box<dyn Chunker>` (what we store in MCP/watch state) be passed wherever
// `&dyn Chunker` is expected, without manual `&**boxed` at every call site.
impl<T: Chunker + ?Sized> Chunker for Box<T> {
    fn chunk(&self, text: &str, rel_path: &str) -> Vec<Chunk> {
        (**self).chunk(text, rel_path)
    }
    fn edges(&self, text: &str, rel_path: &str, chunks: &[Chunk]) -> Vec<EdgeSpec> {
        (**self).edges(text, rel_path, chunks)
    }
}

/// Pick a chunker by resolved source mode. Falls back to markdown for any mode that doesn't
/// have a code chunker yet (sub-slice B will add code).
pub fn from_config(cfg: &Config, _source_root: &Path) -> Box<dyn Chunker> {
    // cfg.source.mode is already resolved (never "auto") by config::resolve.
    let mode = Mode::parse(&cfg.source.mode).unwrap_or(Mode::Notes);
    match mode {
        Mode::Code => Box::new(code::CodeChunker::new(&cfg.chunking)),
        Mode::ClaudeCode => Box::new(claude_code::ClaudeCodeChunker::new(&cfg.claude_code)),
        Mode::Codex => Box::new(codex::CodexChunker::new(&cfg.codex)),
        Mode::Obsidian | Mode::Notes | Mode::Docs | Mode::Auto => {
            Box::new(markdown::MarkdownChunker::from_config(&cfg.chunking))
        }
    }
}

/// What gets fed to the embedder for a given chunk. Anchor (path + heading) + content so the
/// embedder sees both location signal and the actual text.
pub fn embedded_text(rel_path_no_ext: &str, heading_path: &str, content: &str) -> String {
    if heading_path.is_empty() {
        format!("{rel_path_no_ext}\n\n{content}")
    } else {
        format!("{rel_path_no_ext}\n{heading_path}\n\n{content}")
    }
}

/// Lightweight aliases for code symbols. These are search-only hints: canonical
/// `chunks.symbol` stays unchanged for graph resolution and exact lookups.
pub fn symbol_alias_text(heading_path: &str, symbol: &str) -> String {
    let mut aliases = Vec::new();
    push_unique(&mut aliases, symbol.to_string());
    let words = split_identifier(symbol);
    if words.len() > 1 {
        push_unique(&mut aliases, words.join(" "));
        push_unique(&mut aliases, words.join("_"));
        push_unique(&mut aliases, words.join("-"));
    }
    if !heading_path.is_empty() {
        push_unique(&mut aliases, format!("{heading_path}::{symbol}"));
        push_unique(&mut aliases, format!("{heading_path}.{symbol}"));
        let normalized = heading_path.replace("::", ".").replace('/', ".");
        push_unique(&mut aliases, format!("{normalized}.{symbol}"));
    }
    aliases.join("\n")
}

pub fn symbol_matches_alias(symbol: &str, query: &str) -> bool {
    let q = query.trim();
    if q == symbol {
        return true;
    }
    symbol_alias_text("", symbol)
        .lines()
        .any(|alias| alias.eq_ignore_ascii_case(q))
}

fn push_unique(out: &mut Vec<String>, value: String) {
    if !value.trim().is_empty() && !out.iter().any(|v| v == &value) {
        out.push(value);
    }
}

fn split_identifier(input: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    let mut prev: Option<char> = None;
    let chars: Vec<char> = input.chars().collect();
    for (idx, ch) in chars.iter().copied().enumerate() {
        if ch == '_' || ch == '-' || ch == ':' || ch == '.' || ch == '/' {
            flush_word(&mut current, &mut words);
            prev = None;
            continue;
        }
        let next = chars.get(idx + 1).copied();
        let boundary = prev
            .map(|p| {
                (p.is_lowercase() && ch.is_uppercase())
                    || (p.is_alphabetic() && ch.is_numeric())
                    || (p.is_numeric() && ch.is_alphabetic())
                    || (p.is_uppercase()
                        && ch.is_uppercase()
                        && next.map(|n| n.is_lowercase()).unwrap_or(false))
            })
            .unwrap_or(false);
        if boundary {
            flush_word(&mut current, &mut words);
        }
        current.push(ch.to_ascii_lowercase());
        prev = Some(ch);
    }
    flush_word(&mut current, &mut words);
    words
}

fn flush_word(current: &mut String, words: &mut Vec<String>) {
    if !current.is_empty() {
        words.push(std::mem::take(current));
    }
}

#[cfg(test)]
mod alias_tests {
    use super::{split_identifier, symbol_alias_text, symbol_matches_alias};

    #[test]
    fn aliases_split_common_identifier_shapes() {
        assert_eq!(
            split_identifier("processRequest"),
            vec!["process", "request"]
        );
        assert_eq!(
            split_identifier("MAX_RETRY_COUNT"),
            vec!["max", "retry", "count"]
        );
        assert_eq!(split_identifier("HTTPServer2"), vec!["http", "server", "2"]);
    }

    #[test]
    fn aliases_include_qualified_forms() {
        let aliases = symbol_alias_text("Store", "openConnection");
        assert!(aliases.lines().any(|a| a == "open connection"));
        assert!(aliases.lines().any(|a| a == "open_connection"));
        assert!(aliases.lines().any(|a| a == "Store::openConnection"));
        assert!(aliases.lines().any(|a| a == "Store.openConnection"));
        assert!(symbol_matches_alias("openConnection", "open connection"));
    }
}

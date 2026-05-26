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
    Extends,
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

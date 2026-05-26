//! Source `mode` — a single preset that picks chunker + embedder + ignore-dirs + extensions.
//!
//! The point: most users say `--mode obsidian` or `--mode code` and never touch the low-level
//! knobs in `[chunking]` / `[embedder]` / `[vault]`. Explicit overrides in those sections still
//! win when present. Auto-detection is the default: `.obsidian/` directory → `obsidian`,
//! code-extension majority → `code`, `.md` majority → `notes`.

use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::config::{ChunkingConfig, EmbedderConfig, VaultConfig};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Mode {
    Obsidian,
    Notes,
    Docs,
    Code,
    /// Source-specific mode for indexing Claude Code session transcripts
    /// (`~/.claude/projects/<encoded-cwd>/<session>.jsonl`). Other agents (Codex, Aider, etc.)
    /// would get their own mode if/when added — JSONL shapes differ per tool.
    ClaudeCode,
    /// Source-specific mode for indexing OpenAI Codex CLI session transcripts
    /// (`~/.codex/sessions/YYYY/MM/DD/rollout-<iso>-<uuid>.jsonl`). Envelope shape +
    /// function_call/output records differ from claude-code, hence a peer mode.
    Codex,
    /// Resolved at indexing time by `Mode::detect`.
    Auto,
}

impl Default for Mode {
    fn default() -> Self {
        Mode::Auto
    }
}

impl Mode {
    /// Parse a string from config / CLI. Returns `Auto` for unknown values (with a stderr
    /// warning surfaced at the call site, not here — this fn stays pure).
    pub fn parse(s: &str) -> Option<Mode> {
        match s.trim().to_ascii_lowercase().as_str() {
            "obsidian" => Some(Mode::Obsidian),
            "notes" => Some(Mode::Notes),
            "docs" => Some(Mode::Docs),
            "code" => Some(Mode::Code),
            "claude-code" | "claude_code" => Some(Mode::ClaudeCode),
            "codex" => Some(Mode::Codex),
            "auto" | "" => Some(Mode::Auto),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Mode::Obsidian => "obsidian",
            Mode::Notes => "notes",
            Mode::Docs => "docs",
            Mode::Code => "code",
            Mode::ClaudeCode => "claude-code",
            Mode::Codex => "codex",
            Mode::Auto => "auto",
        }
    }

    /// Resolve a config string (may be `None`, `Some("auto")`, or `Some("code")`) against the
    /// source directory. Returns the concrete (non-`Auto`) mode.
    pub fn resolve(configured: &Option<String>, source_root: &Path) -> Mode {
        let parsed = configured
            .as_deref()
            .and_then(Self::parse)
            .unwrap_or(Mode::Auto);
        match parsed {
            Mode::Auto => Self::detect(source_root),
            other => other,
        }
    }

    /// Auto-detect mode by inspecting the source directory. Order:
    ///   1. Path ends with `.claude/projects` → `claude-code` (cheap path-shape check first)
    ///   2. `.obsidian/` present → `obsidian`
    ///   3. Code-extension files outnumber `.md` ≥ 2:1 → `code`
    ///   4. `.md` files outnumber code-extension files (or both zero) → `notes`
    pub fn detect(source_root: &Path) -> Mode {
        if is_claude_code_projects_dir(source_root) {
            return Mode::ClaudeCode;
        }
        if is_codex_sessions_dir(source_root) {
            return Mode::Codex;
        }
        if source_root.join(".obsidian").is_dir() {
            return Mode::Obsidian;
        }
        let DetectCounts { code, md } = count_extensions(source_root);
        if code == 0 && md == 0 {
            return Mode::Notes;
        }
        if code >= md * 2 {
            Mode::Code
        } else {
            Mode::Notes
        }
    }

    pub fn chunking_defaults(&self) -> ChunkingConfig {
        match self {
            Mode::Docs => ChunkingConfig {
                target_bytes: 1200,
                atomic_below_bytes: 1000,
                overlap_bytes: 180,
            },
            // Obsidian/Notes/Auto → adaptive-markdown defaults
            // Code → tree-sitter is structural, the size knobs barely apply; keep defaults
            //   so config layering stays predictable
            _ => ChunkingConfig {
                target_bytes: 1800,
                atomic_below_bytes: 1600,
                overlap_bytes: 270,
            },
        }
    }

    pub fn embedder_defaults(&self) -> EmbedderConfig {
        match self {
            Mode::Code => EmbedderConfig {
                provider: "fastembed".into(),
                model: "jina-embeddings-v2-base-code".into(),
                api_key_env: "OPENAI_API_KEY".into(),
                dimensions: None,
            },
            // ClaudeCode transcripts are prose (the user's natural-language prompts +
            // assistant text), not code — use the prose embedder.
            _ => EmbedderConfig {
                provider: "fastembed".into(),
                model: "bge-small-en-v1.5".into(),
                api_key_env: "OPENAI_API_KEY".into(),
                dimensions: None,
            },
        }
    }

    pub fn vault_defaults(&self) -> VaultConfig {
        let mut ignore: Vec<String> = vec![
            ".dora".into(),
            ".git".into(),
            "node_modules".into(),
        ];
        match self {
            Mode::Obsidian => {
                ignore.push(".obsidian".into());
                ignore.push(".trash".into());
            }
            Mode::Docs => {
                ignore.push("build".into());
                ignore.push("site".into());
                ignore.push("_build".into());
                ignore.push("dist".into());
            }
            Mode::Code => {
                ignore.push("target".into());
                ignore.push("dist".into());
                ignore.push("build".into());
                ignore.push(".venv".into());
                ignore.push("__pycache__".into());
                ignore.push(".next".into());
                ignore.push(".turbo".into());
            }
            Mode::ClaudeCode => {
                // ~/.claude/projects/ is a flat dir of <encoded-project>/*.jsonl folders.
                // No subdirs to ignore beyond the global ones.
            }
            Mode::Codex => {
                // ~/.codex/sessions/ is date-partitioned (YYYY/MM/DD). Nothing extra to ignore.
            }
            Mode::Notes | Mode::Auto => {}
        }
        VaultConfig { ignore }
    }

    /// File extensions this mode walks. The walker uses these to filter; code-mode will use
    /// the language registry (sub-slice B) instead — but we still expose extensions here so
    /// `vault::list_entries` can short-circuit per mode.
    pub fn extensions(&self) -> &'static [&'static str] {
        match self {
            Mode::Obsidian | Mode::Notes => &["md"],
            Mode::Docs => &["md", "mdx", "rst"],
            Mode::Code => &["rs", "py", "pyi", "ts", "tsx", "js", "jsx", "go", "java", "rb"],
            Mode::ClaudeCode | Mode::Codex => &["jsonl"],
            Mode::Auto => &["md"], // shouldn't be reached after resolve()
        }
    }

    /// True if this mode indexes an agent-transcript source (Claude Code, Codex, future).
    /// Callers (the indexer's settle filter, doctor reporting) use it to apply transcript-
    /// specific behavior uniformly without enumerating every variant.
    pub fn is_transcript(&self) -> bool {
        matches!(self, Mode::ClaudeCode | Mode::Codex)
    }

    /// How long a transcript file must have been at-rest before we index it. Active sessions
    /// (the JSONL the agent is currently writing) are skipped. Non-transcript modes return 0
    /// (the filter is a no-op for them).
    pub fn settle_seconds(&self, cfg: &crate::config::Config) -> u64 {
        match self {
            Mode::ClaudeCode => cfg.claude_code.settle_seconds,
            Mode::Codex => cfg.codex.settle_seconds,
            _ => 0,
        }
    }
}

/// True if the path ends with `.claude/projects` (the canonical Claude Code session dir).
fn is_claude_code_projects_dir(p: &Path) -> bool {
    let components: Vec<_> = p
        .components()
        .map(|c| c.as_os_str().to_string_lossy().to_string())
        .collect();
    let n = components.len();
    n >= 2 && components[n - 1] == "projects" && components[n - 2] == ".claude"
}

/// True if the path ends with `.codex/sessions` (the canonical Codex CLI session dir).
fn is_codex_sessions_dir(p: &Path) -> bool {
    let components: Vec<_> = p
        .components()
        .map(|c| c.as_os_str().to_string_lossy().to_string())
        .collect();
    let n = components.len();
    n >= 2 && components[n - 1] == "sessions" && components[n - 2] == ".codex"
}

// ---------- detect-helpers ----------

struct DetectCounts {
    code: usize,
    md: usize,
}

const CODE_EXTS: &[&str] = &["rs", "py", "pyi", "ts", "tsx", "js", "jsx", "go", "java", "rb"];
const MD_EXTS: &[&str] = &["md", "mdx", "rst"];

fn count_extensions(root: &Path) -> DetectCounts {
    let mut code = 0usize;
    let mut md = 0usize;
    let walker = walkdir::WalkDir::new(root)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| {
            if e.file_type().is_dir() {
                let name = e.file_name().to_string_lossy();
                !["target", "node_modules", ".git", ".venv", "dist", "build", ".dora"]
                    .contains(&name.as_ref())
            } else {
                true
            }
        });
    for entry in walker.flatten() {
        if !entry.file_type().is_file() {
            continue;
        }
        let Some(ext) = entry.path().extension().and_then(|s| s.to_str()) else {
            continue;
        };
        if CODE_EXTS.contains(&ext) {
            code += 1;
        } else if MD_EXTS.contains(&ext) {
            md += 1;
        }
    }
    DetectCounts { code, md }
}

/// Human-friendly description of what `detect` saw — used by `dora source add` to print why
/// a particular mode was chosen.
pub fn detection_summary(source_root: &Path) -> String {
    if is_claude_code_projects_dir(source_root) {
        return "path ends with .claude/projects".to_string();
    }
    if is_codex_sessions_dir(source_root) {
        return "path ends with .codex/sessions".to_string();
    }
    if source_root.join(".obsidian").is_dir() {
        return "`.obsidian/` directory present".to_string();
    }
    let DetectCounts { code, md } = count_extensions(source_root);
    format!("{md} .md files, {code} code-extension files")
}

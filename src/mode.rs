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
#[serde(rename_all = "lowercase")]
pub enum Mode {
    Obsidian,
    Notes,
    Docs,
    Code,
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
    ///   1. `.obsidian/` present → `obsidian`
    ///   2. Code-extension files outnumber `.md` ≥ 2:1 → `code`
    ///   3. `.md` files outnumber code-extension files (or both zero) → `notes`
    pub fn detect(source_root: &Path) -> Mode {
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
            Mode::Auto => &["md"], // shouldn't be reached after resolve()
        }
    }
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
    if source_root.join(".obsidian").is_dir() {
        return "`.obsidian/` directory present".to_string();
    }
    let DetectCounts { code, md } = count_extensions(source_root);
    format!("{md} .md files, {code} code-extension files")
}

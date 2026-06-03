//! Centralized on-disk layout for dora's per-source data.
//!
//! Everything dora persists for a source lives under `~/.dora` (override via `$DORA_HOME`),
//! NOT inside the indexed folder. This keeps the user's vault/repo free of any `.dora/` dir
//! (no accidental git commits, no cloud-sync corrupting the index) and lets every source share
//! one model cache instead of duplicating ~33–150 MB of weights per source. Sources are keyed
//! by their unique registered name.
//!
//! ```text
//! ~/.dora/
//! ├── sources/
//! │   └── <name>/
//! │       ├── index.db        per-source SQLite index (kept separate, never merged)
//! │       └── config.toml      per-source mode / embedder / chunking overrides
//! ├── models/                  SHARED embedder cache (fastembed's HF layout dedups by repo)
//! └── source-roots             denormalized newline-separated canonical source paths,
//!                              read by the shell wrappers for fast, dora-free discovery
//! ```
//!
//! Pre-0.9 installs stored all of this co-located at `<source>/.dora/`. The migration path
//! (`migrate_source_if_legacy` in main.rs) moves it out on first touch; [`legacy_dir`] is the
//! only place that still references the old location.

use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};

/// Override the dora data root (mainly for tests; mirrors `DORA_REGISTRY`).
const ENV_HOME: &str = "DORA_HOME";

/// `$DORA_HOME` or `~/.dora`.
pub fn dora_home() -> Result<PathBuf> {
    if let Ok(p) = std::env::var(ENV_HOME) {
        return Ok(PathBuf::from(p));
    }
    let home = dirs::home_dir().ok_or_else(|| anyhow!("could not determine $HOME"))?;
    Ok(home.join(".dora"))
}

/// `~/.dora/sources`.
pub fn sources_root() -> Result<PathBuf> {
    Ok(dora_home()?.join("sources"))
}

/// `~/.dora/sources/<name>` — per-source store dir holding `index.db` + `config.toml`.
pub fn source_store_dir(name: &str) -> Result<PathBuf> {
    Ok(sources_root()?.join(name))
}

/// `~/.dora/sources/<name>/index.db`.
pub fn db_path(name: &str) -> Result<PathBuf> {
    Ok(source_store_dir(name)?.join("index.db"))
}

/// `~/.dora/sources/<name>/config.toml`.
pub fn config_path(name: &str) -> Result<PathBuf> {
    Ok(source_store_dir(name)?.join("config.toml"))
}

/// `~/.dora/models` — the shared embedder cache. Name-independent: every source on the same
/// model resolves to the same `models--<org>--<name>/` subtree, so pointing all sources here
/// is automatically dedup-safe with no custom merge logic.
pub fn models_root() -> Result<PathBuf> {
    Ok(dora_home()?.join("models"))
}

/// `~/.dora/source-roots` — denormalized list of canonical source paths, one per line. Written
/// by `Registry::save()` and consumed by the zsh wrappers (a `grep -qxF` membership check) so
/// flagless `grep` can decide "am I inside a dora source?" without spawning dora or parsing TOML.
pub fn source_roots_file() -> Result<PathBuf> {
    Ok(dora_home()?.join("source-roots"))
}

/// The legacy co-located store dir (`<source_root>/.dora`). Only the migration path should
/// reference this — everything else uses the centralized helpers above.
pub fn legacy_dir(source_root: &Path) -> PathBuf {
    source_root.join(".dora")
}

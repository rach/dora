//! Global dora settings at `~/.config/dora/config.toml`. Distinct from the per-source
//! `<source>/.dora/config.toml` (which carries mode/embedder/chunking per source) and from
//! `~/.config/dora/registry.toml` (the sources list). One concern per file.
//!
//! Today the only setting is `[wrappers] enabled` — controls whether the shell wrappers
//! (`grep` / `rg` / `ag` / `find` injected by `dora install`) route flagless calls into dora
//! or fall through to the real tool. Default is enabled; missing file = enabled.
//!
//! The wrapper hot-path doesn't deserialize this file — it does a one-line `grep` against
//! `^enabled = false` since process spawn for a real TOML parse would dominate every shell
//! invocation. This module is for the CLI side (`dora wrappers <on|off|status>` and doctor).

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

const SETTINGS_FILENAME: &str = "config.toml";

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct Settings {
    pub wrappers: WrappersSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WrappersSettings {
    /// When false, dora's shell wrappers pass through to the real tool (`grep`/`rg`/etc.).
    /// Default true. Missing config file is treated as true.
    pub enabled: bool,
}

impl Default for WrappersSettings {
    fn default() -> Self {
        Self { enabled: true }
    }
}

impl Settings {
    pub fn load() -> Result<Self> {
        let path = settings_path()?;
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("read settings {}", path.display()))?;
        let s: Settings = toml::from_str(&text)
            .with_context(|| format!("parse settings {}", path.display()))?;
        Ok(s)
    }

    pub fn save(&self) -> Result<()> {
        let path = settings_path()?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create config dir {}", parent.display()))?;
        }
        let tmp = path.with_extension("toml.tmp");
        let text = toml::to_string_pretty(self).context("serialize settings")?;
        std::fs::write(&tmp, text).with_context(|| format!("write tmp {}", tmp.display()))?;
        std::fs::rename(&tmp, &path)
            .with_context(|| format!("rename {} -> {}", tmp.display(), path.display()))?;
        Ok(())
    }
}

pub fn settings_path() -> Result<PathBuf> {
    let home = dirs::home_dir()
        .ok_or_else(|| anyhow::anyhow!("could not determine $HOME"))?;
    Ok(home.join(".config").join("dora").join(SETTINGS_FILENAME))
}

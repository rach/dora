//! Global source registry at `~/.config/dora/registry.toml`. Tracks the set of indexed
//! directories that `dora mcp` (multi-source mode) serves. Each entry has a unique name and
//! an absolute path. The vault concept ("a directory of markdown") is one *kind* of source —
//! the registry abstracts over any indexed directory: notes, code, transcripts, etc.

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

const REGISTRY_FILENAME: &str = "registry.toml";

/// Override the registry location (mainly for tests + transient diagnostics).
const ENV_REGISTRY: &str = "DORA_REGISTRY";

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct Registry {
    #[serde(rename = "source", default)]
    pub sources: Vec<Source>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    pub name: String,
    pub path: PathBuf,
    /// Free-form one-paragraph description of what this source contains. Surfaced to agents
    /// via the MCP `list_sources` tool and embedded into the `search` tool's input schema
    /// so the agent can pick the right `source` without an extra round-trip.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl Registry {
    /// Read the registry from disk, or return an empty one if the file doesn't exist.
    pub fn load() -> Result<Self> {
        let path = registry_path()?;
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("read registry {}", path.display()))?;
        let reg: Registry =
            toml::from_str(&text).with_context(|| format!("parse registry {}", path.display()))?;
        Ok(reg)
    }

    /// Atomically write registry to disk (write to tmp + rename).
    pub fn save(&self) -> Result<()> {
        let path = registry_path()?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create config dir {}", parent.display()))?;
        }
        let tmp = path.with_extension("toml.tmp");
        let text = toml::to_string_pretty(self).context("serialize registry")?;
        std::fs::write(&tmp, text).with_context(|| format!("write tmp {}", tmp.display()))?;
        std::fs::rename(&tmp, &path)
            .with_context(|| format!("rename {} -> {}", tmp.display(), path.display()))?;
        Ok(())
    }

    pub fn find_by_name(&self, name: &str) -> Option<&Source> {
        self.sources.iter().find(|s| s.name == name)
    }

    pub fn find_by_path(&self, path: &Path) -> Option<&Source> {
        let target = path.canonicalize().ok()?;
        self.sources
            .iter()
            .find(|s| s.path.canonicalize().ok() == Some(target.clone()))
    }

    pub fn add(&mut self, source: Source) -> Result<()> {
        if self.find_by_name(&source.name).is_some() {
            bail!(
                "source name '{}' already registered. pick a different --name.",
                source.name
            );
        }
        if self.sources.iter().any(|s| s.path == source.path) {
            bail!(
                "path {} is already registered as '{}'",
                source.path.display(),
                self.sources
                    .iter()
                    .find(|s| s.path == source.path)
                    .map(|s| s.name.as_str())
                    .unwrap_or("?")
            );
        }
        self.sources.push(source);
        Ok(())
    }

    pub fn remove(&mut self, name: &str) -> Result<Source> {
        let idx = self
            .sources
            .iter()
            .position(|s| s.name == name)
            .ok_or_else(|| anyhow!("no source named '{name}'"))?;
        Ok(self.sources.remove(idx))
    }

    /// Update the description of an existing source. Errors if not found.
    pub fn set_description(&mut self, name: &str, description: Option<String>) -> Result<()> {
        let s = self
            .sources
            .iter_mut()
            .find(|s| s.name == name)
            .ok_or_else(|| anyhow!("no source named '{name}'"))?;
        s.description = description;
        Ok(())
    }
}

/// Resolution order: `$DORA_REGISTRY` env var → `$HOME/.config/dora/registry.toml`.
/// We use the XDG-style path explicitly (not `dirs::config_dir()`) so the location is the
/// same on macOS and Linux — matches README + what most personal-tool users expect.
pub fn registry_path() -> Result<PathBuf> {
    if let Ok(p) = std::env::var(ENV_REGISTRY) {
        return Ok(PathBuf::from(p));
    }
    let home = dirs::home_dir()
        .ok_or_else(|| anyhow!("could not determine $HOME"))?;
    Ok(home.join(".config").join("dora").join(REGISTRY_FILENAME))
}

/// Convenience for `cmd_search` — look up the registered name (if any) for a given path.
/// Returns None if the registry is missing/empty or the path isn't registered.
pub fn find_source_name_for_path(path: &Path) -> Option<String> {
    let reg = Registry::load().ok()?;
    reg.find_by_path(path).map(|s| s.name.clone())
}

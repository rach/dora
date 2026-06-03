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

    /// Atomically write registry to disk (write to tmp + rename), then regenerate the
    /// denormalized `~/.dora/source-roots` file the shell wrappers read.
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
        // Best-effort: keep the wrapper's discovery file in sync. A failure here (e.g. odd
        // perms on ~/.dora) shouldn't fail the registry write itself.
        let _ = self.write_roots_file();
        Ok(())
    }

    /// Write `~/.dora/source-roots`: one canonical source path per line. The zsh wrappers walk
    /// up from cwd and `grep -qxF` each ancestor against this file to decide "am I inside a
    /// dora source?" without spawning dora or parsing TOML. Paths that fail to canonicalize
    /// (e.g. a source whose folder was deleted) are skipped.
    pub fn write_roots_file(&self) -> Result<()> {
        let path = crate::paths::source_roots_file()?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create dora dir {}", parent.display()))?;
        }
        let mut body = String::new();
        for s in &self.sources {
            let canon = s.path.canonicalize().unwrap_or_else(|_| s.path.clone());
            body.push_str(&canon.to_string_lossy());
            body.push('\n');
        }
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, body).with_context(|| format!("write tmp {}", tmp.display()))?;
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

    /// Map an arbitrary path (possibly a deep subfolder) to the source that owns it, by
    /// longest canonical-prefix match. With storage moved out of the folder, this is how
    /// `dora "query"` and the read commands locate the index from any working directory.
    /// When sources nest, the most specific (longest path) wins.
    pub fn resolve_for_path(&self, path: &Path) -> Option<&Source> {
        let target = path.canonicalize().ok()?;
        self.sources
            .iter()
            .filter(|s| {
                s.path
                    .canonicalize()
                    .ok()
                    .map(|sp| target == sp || target.starts_with(&sp))
                    .unwrap_or(false)
            })
            .max_by_key(|s| {
                s.path
                    .canonicalize()
                    .map(|p| p.components().count())
                    .unwrap_or(0)
            })
    }

    pub fn add(&mut self, source: Source) -> Result<()> {
        validate_source_name(&source.name)?;
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

    /// Rename a source in the registry. Validates the new name and rejects collisions. The
    /// on-disk store dir (`~/.dora/sources/<name>`) is moved by the caller (`cmd_source`).
    pub fn rename(&mut self, old: &str, new: &str) -> Result<()> {
        validate_source_name(new)?;
        if self.find_by_name(new).is_some() {
            bail!("source name '{new}' already registered");
        }
        let s = self
            .sources
            .iter_mut()
            .find(|s| s.name == old)
            .ok_or_else(|| anyhow!("no source named '{old}'"))?;
        s.name = new.to_string();
        Ok(())
    }
}

/// Validate that a source name is safe to use as a single filesystem directory component
/// (it becomes `~/.dora/sources/<name>`). Allows `[A-Za-z0-9._-]`, rejects empties, `.`/`..`,
/// path separators, and anything over 64 chars.
pub fn validate_source_name(name: &str) -> Result<()> {
    if name.is_empty() {
        bail!("source name must not be empty");
    }
    if name.len() > 64 {
        bail!("source name '{name}' is too long (max 64 chars)");
    }
    if name == "." || name == ".." {
        bail!("source name must not be '.' or '..'");
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
    {
        bail!("source name '{name}' has invalid characters (allowed: letters, digits, . _ -)");
    }
    Ok(())
}

/// Resolution order: `$DORA_REGISTRY` env var → `$HOME/.config/dora/registry.toml`.
/// We use the XDG-style path explicitly (not `dirs::config_dir()`) so the location is the
/// same on macOS and Linux — matches README + what most personal-tool users expect.
pub fn registry_path() -> Result<PathBuf> {
    if let Ok(p) = std::env::var(ENV_REGISTRY) {
        return Ok(PathBuf::from(p));
    }
    let home = dirs::home_dir().ok_or_else(|| anyhow!("could not determine $HOME"))?;
    Ok(home.join(".config").join("dora").join(REGISTRY_FILENAME))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn src(name: &str, path: &Path) -> Source {
        Source {
            name: name.into(),
            path: path.to_path_buf(),
            description: None,
        }
    }

    #[test]
    fn validate_names() {
        assert!(validate_source_name("brain").is_ok());
        assert!(validate_source_name("my-notes_2.0").is_ok());
        assert!(validate_source_name("").is_err());
        assert!(validate_source_name(".").is_err());
        assert!(validate_source_name("..").is_err());
        assert!(validate_source_name("a/b").is_err()); // path separator
        assert!(validate_source_name("a b").is_err()); // space
        assert!(validate_source_name(&"x".repeat(65)).is_err()); // too long
    }

    #[test]
    fn resolve_for_path_longest_prefix() {
        let parent = tempdir().unwrap();
        let child = parent.path().join("nested");
        std::fs::create_dir_all(&child).unwrap();
        let deep = child.join("a").join("b");
        std::fs::create_dir_all(&deep).unwrap();
        let other = parent.path().join("other");
        std::fs::create_dir_all(&other).unwrap();

        let reg = Registry {
            sources: vec![src("parent", parent.path()), src("child", &child)],
        };

        // exact match
        assert_eq!(reg.resolve_for_path(parent.path()).unwrap().name, "parent");
        // deep subfolder owned by both → most specific (child) wins
        assert_eq!(reg.resolve_for_path(&deep).unwrap().name, "child");
        // subfolder only under parent
        assert_eq!(reg.resolve_for_path(&other).unwrap().name, "parent");
        // unrelated path → none
        let unrelated = tempdir().unwrap();
        assert!(reg.resolve_for_path(unrelated.path()).is_none());
    }

    #[test]
    fn rename_rules() {
        let mut reg = Registry {
            sources: vec![src("a", Path::new("/tmp/a")), src("b", Path::new("/tmp/b"))],
        };
        assert!(reg.rename("a", "c").is_ok());
        assert_eq!(reg.find_by_name("c").unwrap().path, PathBuf::from("/tmp/a"));
        assert!(reg.find_by_name("a").is_none());
        assert!(reg.rename("c", "b").is_err()); // target name taken
        assert!(reg.rename("missing", "z").is_err()); // unknown source
        assert!(reg.rename("c", "bad name").is_err()); // invalid new name
    }
}

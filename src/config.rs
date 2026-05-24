//! User-facing TOML config at `<vault>/.dora/config.toml`.
//!
//! Every section + every key is optional. Missing file → all defaults. Defaults match the POC's
//! hardcoded constants exactly, so an existing user with no config gets the same behavior.

use anyhow::Result;
use serde::Deserialize;
use std::path::Path;

/// Bumped whenever schema changes in a way old data can't read. Mismatch → require fresh index.
/// v0 item B added mtime/size/content_hash columns to `files`; bumping forces a one-time rebuild.
pub const SCHEMA_VERSION: &str = "2";

/// Bumped whenever the chunking algorithm changes in a way that would produce different chunks
/// for the same input. Mismatch → drop chunks + reindex.
pub const CHUNKER_VERSION: &str = "1";

#[derive(Debug, Deserialize, Default, Clone)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub vault: VaultConfig,
    pub chunking: ChunkingConfig,
    pub search: SearchConfig,
    pub embedder: EmbedderConfig,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(default, deny_unknown_fields)]
pub struct VaultConfig {
    pub ignore: Vec<String>,
}

impl Default for VaultConfig {
    fn default() -> Self {
        Self {
            ignore: vec![
                ".obsidian".into(),
                ".git".into(),
                ".dora".into(),
                "node_modules".into(),
            ],
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
#[serde(default, deny_unknown_fields)]
pub struct ChunkingConfig {
    pub target_bytes: usize,
    pub atomic_below_bytes: usize,
    pub overlap_bytes: usize,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            target_bytes: 1800,
            atomic_below_bytes: 1600,
            overlap_bytes: 270,
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
#[serde(default, deny_unknown_fields)]
pub struct SearchConfig {
    pub top_k: usize,
    pub collapse_per_file: bool,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            top_k: 10,
            collapse_per_file: true,
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
#[serde(default, deny_unknown_fields)]
pub struct EmbedderConfig {
    /// `"fastembed"` (default) or `"openai"`.
    pub provider: String,
    /// Model name. For fastembed: short name or full HF path (resolved in `embed::fastembed`).
    /// For openai: one of `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002`.
    pub model: String,
    /// Name of the env var holding the provider API key. Default `"OPENAI_API_KEY"`. Only
    /// consulted by remote providers.
    pub api_key_env: String,
    /// Optional vector dimension override. Only supported by openai's text-embedding-3-*. When
    /// set it participates in the canonical embedder id (`openai:text-embedding-3-small:dims=512`)
    /// so changing it triggers a clean reindex.
    pub dimensions: Option<usize>,
}

impl Default for EmbedderConfig {
    fn default() -> Self {
        Self {
            provider: "fastembed".into(),
            model: "bge-small-en-v1.5".into(),
            api_key_env: "OPENAI_API_KEY".into(),
            dimensions: None,
        }
    }
}

impl Config {
    pub fn load_or_default(vault: &Path) -> Result<Self> {
        let path = vault.join(".dora").join("config.toml");
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(&path)?;
        let cfg: Config = toml::from_str(&text)?;
        Ok(cfg)
    }
}

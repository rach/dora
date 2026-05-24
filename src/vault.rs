//! Vault filesystem access. Today: local FS. Shaped so a future Canon trait (S3, etc.) can
//! carve out exactly these two operations — `list_entries` is the metadata-only LIST analog,
//! `read_file` is the GetObject analog. Diff loops elsewhere talk to vault.rs through this
//! surface only; no direct walkdir / fs::read_to_string anywhere else in the codebase.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// Metadata for a single vault file. Returned by `list_entries` without reading the body.
#[derive(Debug, Clone)]
pub struct EntryStat {
    pub relative_path: PathBuf,
    pub mtime: u64,
    pub size: u64,
}

/// Walk the vault and return metadata for every `.md` file. Body bytes are NOT read.
/// Maps to S3 `ListObjectsV2` when remote canon eventually arrives.
pub fn list_entries(root: &Path, ignore_dirs: &[String]) -> Result<Vec<EntryStat>> {
    let root = root.canonicalize().context("canonicalize vault root")?;
    let mut out = Vec::new();

    let walker = WalkDir::new(&root)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| {
            if e.file_type().is_dir() {
                let name = e.file_name().to_string_lossy();
                !ignore_dirs.iter().any(|d| d == name.as_ref())
            } else {
                true
            }
        });

    for entry in walker {
        let entry = match entry {
            Ok(e) => e,
            Err(err) => {
                eprintln!("walk: skipping entry: {err}");
                continue;
            }
        };
        if !entry.file_type().is_file() {
            continue;
        }
        if entry.path().extension().and_then(|s| s.to_str()) != Some("md") {
            continue;
        }

        let meta = match entry.metadata() {
            Ok(m) => m,
            Err(err) => {
                eprintln!("walk: skipping {} ({err})", entry.path().display());
                continue;
            }
        };
        let mtime = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let size = meta.len();

        let rel = entry
            .path()
            .strip_prefix(&root)
            .unwrap_or(entry.path())
            .to_path_buf();

        out.push(EntryStat {
            relative_path: rel,
            mtime,
            size,
        });
    }

    Ok(out)
}

/// Read the body of a single vault file as UTF-8. Maps to S3 `GetObject` later.
pub fn read_file(root: &Path, relative_path: &Path) -> Result<String> {
    let full = root.join(relative_path);
    std::fs::read_to_string(&full).with_context(|| format!("read {}", full.display()))
}

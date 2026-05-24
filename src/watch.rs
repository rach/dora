//! `dora watch` — foreground notify-rs watcher. Coalesces bursty file events into per-source
//! debounced incremental walks. Single-threaded (notify uses its own internal OS thread, we
//! just drain a channel here). Ctrl-C shuts down cleanly.

use anyhow::{Context, Result};
use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use crate::chunk::{self, Chunker};
use crate::config::Config;
use crate::embed::{self, DynEmbedder};
use crate::registry::Source;
use crate::store::Store;
use crate::{check_meta, db_path, models_dir, run_incremental_index};

const IGNORE_DIR_NAMES: &[&str] = &[".dora", ".obsidian", ".git", "node_modules"];
const DEBOUNCE_MS: u64 = 500;
const TICK_MS: u64 = 200;

struct SourceCtx {
    name: String,
    root: PathBuf,
    cfg: Config,
    embedder: DynEmbedder,
    chunker: Box<dyn Chunker>,
    store: Store,
}

pub fn run(sources: Vec<Source>) -> Result<()> {
    if sources.is_empty() {
        anyhow::bail!("nothing to watch (registry empty or filtered to zero sources)");
    }

    // Drop the PID into ~/.config/dora/watch.pid so `dora doctor` can detect us robustly.
    // Stale PIDs on crash are fine: doctor verifies liveness before reporting.
    if let Some(pid_path) = pid_file_path() {
        if let Some(parent) = pid_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let _ = std::fs::write(&pid_path, std::process::id().to_string());
    }

    // Build per-source state, sharing embedders by (provider,model,dimensions).
    let mut cache: HashMap<String, DynEmbedder> = HashMap::new();
    let mut ctxs: Vec<SourceCtx> = Vec::new();
    for src in sources {
        match try_load(&src, &mut cache) {
            Ok(ctx) => ctxs.push(ctx),
            Err(e) => eprintln!(
                "dora watch: skipping source '{}' ({}): {e}",
                src.name,
                src.path.display()
            ),
        }
    }
    if ctxs.is_empty() {
        anyhow::bail!("no sources could be loaded — see errors above");
    }

    // Map absolute source-root → name, used to attribute events to a source.
    let roots: Vec<(String, PathBuf)> = ctxs.iter().map(|c| (c.name.clone(), c.root.clone())).collect();

    // notify channel.
    let (tx, rx) = mpsc::channel::<Result<Event, notify::Error>>();
    let mut watcher: RecommendedWatcher = notify::recommended_watcher(move |res| {
        let _ = tx.send(res);
    })
    .context("init notify watcher")?;
    for (_, root) in &roots {
        watcher
            .watch(root, RecursiveMode::Recursive)
            .with_context(|| format!("watch {}", root.display()))?;
    }

    log_ready(&ctxs);

    let mut pending: HashMap<String, Instant> = HashMap::new();
    loop {
        // Drain available events without blocking.
        loop {
            match rx.try_recv() {
                Ok(Ok(event)) => {
                    if let Some(name) = relevant_source(&event, &roots) {
                        pending.insert(name, Instant::now());
                    }
                }
                Ok(Err(e)) => {
                    eprintln!("dora watch: notify error: {e}");
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    anyhow::bail!("notify channel disconnected");
                }
            }
        }

        // Flush any source whose latest event is older than the debounce window.
        let now = Instant::now();
        let debounce = Duration::from_millis(DEBOUNCE_MS);
        let due: Vec<String> = pending
            .iter()
            .filter_map(|(name, last)| {
                if now.duration_since(*last) >= debounce {
                    Some(name.clone())
                } else {
                    None
                }
            })
            .collect();
        for name in due {
            pending.remove(&name);
            let Some(ctx) = ctxs.iter_mut().find(|c| c.name == name) else {
                continue;
            };
            eprintln!("dora watch: refreshing source={}", ctx.name);
            match run_incremental_index(
                &ctx.root,
                &ctx.cfg,
                &ctx.chunker,
                ctx.embedder.as_ref(),
                &mut ctx.store,
                false,
            ) {
                Ok(summary) => eprintln!(
                    "  {} inserted, {} updated, {} touched, {} renamed, {} deleted, {} unchanged",
                    summary.inserted,
                    summary.updated,
                    summary.touched,
                    summary.renamed,
                    summary.deleted,
                    summary.skipped,
                ),
                Err(e) => eprintln!("  refresh failed: {e}"),
            }
        }

        std::thread::sleep(Duration::from_millis(TICK_MS));
    }
}

fn try_load(src: &Source, cache: &mut HashMap<String, DynEmbedder>) -> Result<SourceCtx> {
    let root = src
        .path
        .canonicalize()
        .with_context(|| format!("canonicalize {}", src.path.display()))?;
    let cfg = Config::load_or_default(&root)?;
    let key = embed::cache_key(&cfg.embedder);
    let embedder = match cache.get(&key) {
        Some(e) => e.clone(),
        None => {
            let new = embed::from_config(&cfg.embedder, &models_dir(&root))?;
            cache.insert(key, new.clone());
            new
        }
    };
    let chunker = chunk::from_config(&cfg, &root);

    let db = db_path(&root);
    if !db.exists() {
        anyhow::bail!(".dora/index.db not found — run `dora index {}` first", root.display());
    }
    let store = Store::open(&db, embedder.dims())?;
    check_meta(&store, embedder.as_ref())?;

    Ok(SourceCtx {
        name: src.name.clone(),
        root,
        cfg,
        embedder,
        chunker,
        store,
    })
}

fn relevant_source(event: &Event, roots: &[(String, PathBuf)]) -> Option<String> {
    // Only data-modifying events trigger a refresh — skip access-only events.
    match event.kind {
        EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_) => {}
        _ => return None,
    }
    for path in &event.paths {
        if is_ignored(path) {
            continue;
        }
        // Match path against each registered source root by ancestor walk.
        for (name, root) in roots {
            if path.starts_with(root) {
                return Some(name.clone());
            }
        }
    }
    None
}

fn is_ignored(path: &Path) -> bool {
    path.components().any(|c| {
        let s = c.as_os_str().to_string_lossy();
        IGNORE_DIR_NAMES.contains(&s.as_ref())
    })
}

/// Where `dora watch` records its PID so `dora doctor` can detect it. Cross-platform
/// `$HOME/.config/dora/watch.pid` (matches registry.toml's resolution).
pub fn pid_file_path() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".config").join("dora").join("watch.pid"))
}

fn log_ready(ctxs: &[SourceCtx]) {
    eprintln!("dora watch: watching {} source(s):", ctxs.len());
    for c in ctxs {
        eprintln!("  - {} ({})", c.name, c.root.display());
    }
    eprintln!("(Ctrl-C to stop)");
}

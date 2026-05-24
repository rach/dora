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
use crate::registry::{self, Registry, Source};
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
    // Note: an empty registry is intentionally NOT an error. The watcher subscribes to the
    // registry dir and picks up the first `dora source add` whenever it lands — required for
    // the `brew services start dora` flow, where the user may install + start the service
    // before they've registered anything.

    // Drop the PID into ~/.config/dora/watch.pid so `dora doctor` can detect us robustly.
    // Stale PIDs on crash are fine: doctor verifies liveness before reporting.
    if let Some(pid_path) = pid_file_path() {
        if let Some(parent) = pid_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let _ = std::fs::write(&pid_path, std::process::id().to_string());
    }

    // Build per-source state, sharing embedders by (provider,model,dimensions). `cache`
    // and `roots` live through the whole loop because the registry-reload path mutates
    // both — new sources push into them, removed sources drop out.
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
        eprintln!(
            "dora watch: no sources registered yet — waiting. Register one via `dora source add <path>`."
        );
    }

    // Map absolute source-root → name, used to attribute events to a source.
    let mut roots: Vec<(String, PathBuf)> = ctxs
        .iter()
        .map(|c| (c.name.clone(), c.root.clone()))
        .collect();

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

    // Also subscribe to the registry file's parent dir so newly-added sources are
    // picked up without a restart. Watching the dir (not the file) handles atomic
    // rename writes correctly — the rename surfaces as a Create on the final path.
    // Sibling files (watch.pid, future configs) get filtered out by exact-path match.
    let registry_path = registry::registry_path()
        .context("resolve registry path for reload-watch")?;
    let registry_dir = registry_path
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| anyhow::anyhow!("registry path has no parent"))?;
    if registry_dir.exists() {
        if let Err(e) = watcher.watch(&registry_dir, RecursiveMode::NonRecursive) {
            eprintln!(
                "dora watch: couldn't subscribe to registry dir {} ({e}); auto-reload disabled",
                registry_dir.display()
            );
        }
    }

    log_ready(&ctxs);

    let mut pending: HashMap<String, Instant> = HashMap::new();
    let mut registry_dirty = false;
    loop {
        // Drain available events without blocking.
        loop {
            match rx.try_recv() {
                Ok(Ok(event)) => {
                    // Classify: registry-touch vs source-touch. Only data-modifying events
                    // for the registry trigger a reload — Access events are noise.
                    if matches!(
                        event.kind,
                        EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_)
                    ) && event.paths.iter().any(|p| p == &registry_path)
                    {
                        registry_dirty = true;
                    } else if let Some(name) = relevant_source(&event, &roots) {
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

        if registry_dirty {
            registry_dirty = false;
            if let Err(e) = reload_sources(
                &mut ctxs,
                &mut roots,
                &mut pending,
                &mut cache,
                &mut watcher,
            ) {
                eprintln!("dora watch: registry reload failed: {e}");
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

/// Diff the on-disk registry against the currently-loaded sources, then bring the watcher
/// in sync: add+watch new entries, unwatch+drop removed ones. Errors loading a single new
/// source (missing `.dora/index.db`, embedder mismatch) log + skip — they don't tear down
/// the already-running watcher.
fn reload_sources(
    ctxs: &mut Vec<SourceCtx>,
    roots: &mut Vec<(String, PathBuf)>,
    pending: &mut HashMap<String, Instant>,
    cache: &mut HashMap<String, DynEmbedder>,
    watcher: &mut RecommendedWatcher,
) -> Result<()> {
    let reg = Registry::load().context("reload registry")?;

    let current_names: Vec<String> = ctxs.iter().map(|c| c.name.clone()).collect();
    let desired_names: Vec<String> = reg.sources.iter().map(|s| s.name.clone()).collect();

    // Removals first so a same-name re-add (path change) gets a clean watch.
    for name in &current_names {
        if !desired_names.contains(name) {
            if let Some(idx) = ctxs.iter().position(|c| &c.name == name) {
                let removed = ctxs.remove(idx);
                if let Err(e) = watcher.unwatch(&removed.root) {
                    eprintln!(
                        "dora watch: unwatch {} failed ({e}); continuing",
                        removed.root.display()
                    );
                }
                roots.retain(|(n, _)| n != name);
                pending.remove(name);
                eprintln!("dora watch: - source '{}'", name);
            }
        }
    }

    for src in &reg.sources {
        if current_names.contains(&src.name) {
            continue;
        }
        match try_load(src, cache) {
            Ok(ctx) => {
                let root = ctx.root.clone();
                if let Err(e) = watcher.watch(&root, RecursiveMode::Recursive) {
                    eprintln!(
                        "dora watch: watch {} failed ({e}); source '{}' won't get file events",
                        root.display(),
                        src.name
                    );
                }
                eprintln!("dora watch: + source '{}' ({})", ctx.name, ctx.root.display());
                roots.push((ctx.name.clone(), ctx.root.clone()));
                ctxs.push(ctx);
            }
            Err(e) => eprintln!(
                "dora watch: skipping new source '{}' ({}): {e}",
                src.name,
                src.path.display()
            ),
        }
    }
    Ok(())
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

/// True iff the PID file exists AND the recorded PID responds to `kill -0`. Stale PID
/// files (left behind by a crashed watcher) return false — callers can rely on this for
/// "is a watcher actually running right now?" checks (cmd_source hint, doctor).
pub fn is_running() -> bool {
    let Some(path) = pid_file_path() else {
        return false;
    };
    let Ok(s) = std::fs::read_to_string(&path) else {
        return false;
    };
    let pid_str = s.trim();
    if pid_str.is_empty() {
        return false;
    }
    std::process::Command::new("kill")
        .args(["-0", pid_str])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn log_ready(ctxs: &[SourceCtx]) {
    if ctxs.is_empty() {
        eprintln!("dora watch: ready (no sources yet; will pick up new ones automatically)");
        eprintln!("(Ctrl-C to stop)");
        return;
    }
    eprintln!("dora watch: watching {} source(s):", ctxs.len());
    for c in ctxs {
        eprintln!("  - {} ({})", c.name, c.root.display());
    }
    eprintln!("(Ctrl-C to stop; new sources auto-picked up via `dora source add`)");
}

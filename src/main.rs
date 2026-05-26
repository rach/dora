mod chunk;
mod config;
mod doctor;
mod embed;
mod install;
mod mcp;
mod migrations;
mod mode;
mod pagerank;
mod registry;
mod search;
mod settings;
mod store;
mod vault;
mod watch;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::chunk::Chunker;
type BoxedChunker = Box<dyn Chunker>;
use crate::config::{Config, CHUNKER_VERSION, SCHEMA_VERSION};
use crate::embed::{DynEmbedder, Embedder};
use crate::store::{ChunkRow, Store};

/// If a search-triggered walk would otherwise fire within this many seconds of the last
/// completed walk, skip it. Debounces rapid MCP calls + shell pipelines.
const WALK_DEBOUNCE_SECS: u64 = 2;

#[derive(Parser)]
#[command(
    name = "dora",
    about = "Personal memory: semantic search over a markdown vault.",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    /// Query string (used when no subcommand is given).
    query: Option<String>,

    /// Output JSON instead of ripgrep-style.
    #[arg(long, global = true)]
    json: bool,

    /// Override the configured top_k for this call.
    #[arg(long, global = true)]
    top_k: Option<usize>,

    /// Drop any hit whose merged RRF score is below this threshold. Combines with `--all`
    /// for "give me every relevant doc above this confidence" agentic flows.
    #[arg(long, global = true, value_name = "FLOAT")]
    min_score: Option<f64>,

    /// Disable the top_k cap and return every hit that passed `--min-score` (if set, else
    /// every hit at all). Useful with `--files` to enumerate every matching file.
    #[arg(long, global = true)]
    all: bool,

    /// Files-only output: dedupe hits by path, return path list. Each line is a path; no
    /// `:line:` prefix, no snippet, no heading. Pairs well with `--all` / `--min-score`.
    #[arg(long, global = true)]
    files: bool,
}

#[derive(Subcommand)]
enum Command {
    /// Incremental index of the source at <path> (defaults to current directory).
    Index {
        /// Source root to index. Defaults to current working directory.
        path: Option<PathBuf>,

        /// Walk + diff + (for paid providers) print cost estimate, then exit without embedding.
        #[arg(long)]
        dry_run: bool,
    },
    /// Run a stdio MCP server. By default serves every source in the global registry
    /// (`~/.config/dora/registry.toml`). Use `--source <path>` to serve a single ad-hoc
    /// source instead, or `--include`/`--exclude` to scope to a subset of the registry.
    /// Exposes tools: `search` (with optional `source` parameter) and `list_sources`.
    Mcp {
        /// Serve a single ad-hoc source at this path, ignoring the registry. Mutually
        /// exclusive with --include / --exclude.
        #[arg(long, value_name = "PATH", conflicts_with_all = &["include", "exclude"])]
        source: Option<PathBuf>,

        /// Serve only these registered sources (comma-separated or repeated).
        /// Mutually exclusive with --exclude and --source.
        #[arg(long, value_name = "NAME", value_delimiter = ',', conflicts_with = "exclude")]
        include: Vec<String>,

        /// Serve every registered source EXCEPT these.
        /// Mutually exclusive with --include and --source.
        #[arg(long, value_name = "NAME", value_delimiter = ',')]
        exclude: Vec<String>,
    },
    /// Manage the global source registry.
    Source {
        #[command(subcommand)]
        action: SourceAction,
    },
    /// Auto-patch MCP host configs (Claude Code / Cursor / Codex) and install zsh
    /// wrappers for `grep` (and optionally rg/ag/find). Idempotent. The `--wrap` list is
    /// authoritative — re-running with `--wrap rg` after `--wrap grep,rg` removes the grep block.
    Install {
        /// Only patch a specific client. Default: all detected.
        #[arg(long, value_enum)]
        client: Option<InstallClient>,
        /// Pass `--include <names>` to the patched `dora mcp` block.
        #[arg(long, value_delimiter = ',')]
        include: Vec<String>,
        /// Pass `--exclude <names>` to the patched `dora mcp` block.
        #[arg(long, value_delimiter = ',')]
        exclude: Vec<String>,
        /// Skip the zsh wrapper injection entirely.
        #[arg(long)]
        no_shell: bool,
        /// Comma-separated wrappers to install in ~/.zshrc. Supported: grep, rg, ag, find.
        /// Default: all four. Wrappers are inert when the underlying tool isn't installed —
        /// `command rg "$@"` just errors out same as without us. Any previously-installed
        /// wrapper NOT in this list gets removed (so `--wrap grep` removes rg/ag/find).
        #[arg(
            long,
            value_name = "TOOLS",
            value_delimiter = ',',
            default_value = "grep,rg,ag,find"
        )]
        wrap: Vec<String>,
    },
    /// Report the health of the install: binary, registry, MCP host registration,
    /// shell wrapper, watcher status.
    Doctor,
    /// Foreground file-watcher that keeps registered sources fresh. Runs until Ctrl-C.
    /// Same `--include` / `--exclude` semantics as `dora mcp`.
    Watch {
        #[arg(long, value_name = "NAME", value_delimiter = ',', conflicts_with = "exclude")]
        include: Vec<String>,
        #[arg(long, value_name = "NAME", value_delimiter = ',')]
        exclude: Vec<String>,
    },
    /// Enable / disable the zsh wrappers that route flagless `grep` (and rg/ag/find) into dora.
    Wrappers {
        #[command(subcommand)]
        action: WrappersAction,
    },
    /// Per-subpath context strings — descriptive metadata surfaced alongside search hits.
    /// Use `/` as the prefix for a source-wide default; subtree prefixes override the parent.
    Context {
        #[command(subcommand)]
        action: ContextAction,
    },
}

#[derive(Subcommand)]
enum ContextAction {
    /// Attach a context description to a path prefix within a registered source.
    Add {
        source: String,
        /// Path prefix, e.g. `/api`, `/sdk/typescript`, or `/` for the whole source.
        prefix: String,
        /// Description text shown to agents on hits under this prefix.
        text: String,
    },
    /// List every context registered for a source.
    List { source: String },
    /// Remove a context entry by prefix.
    Remove { source: String, prefix: String },
}

#[derive(Subcommand)]
enum WrappersAction {
    /// Turn the wrappers off — `grep` (etc.) become normal again. Wrappers stay installed
    /// in `~/.zshrc`; they just pass through when this flag is set. Re-enable with
    /// `dora wrappers on`.
    Off,
    /// Turn the wrappers back on (default state).
    On,
    /// Print whether dora's wrappers are currently routing or passing through. With `-q`,
    /// no output is printed; exit code 0 means enabled, 1 means disabled. The wrapper
    /// template in `~/.zshrc` uses the quiet form as its hot-path check.
    Status {
        #[arg(short, long)]
        quiet: bool,
    },
}

#[derive(Clone, clap::ValueEnum)]
enum InstallClient {
    Claude,
    Cursor,
    Codex,
}

#[derive(Subcommand)]
enum SourceAction {
    /// Register an indexed source. Path must already contain `.dora/index.db`.
    Add {
        /// Absolute or relative path to the source root.
        path: PathBuf,
        /// Override the auto-derived name (default: last path component).
        #[arg(long)]
        name: Option<String>,
        /// Free-form description shown to agents in `list_sources` and in `search`'s schema.
        #[arg(long)]
        description: Option<String>,
        /// Indexing mode. `obsidian` / `notes` / `docs` / `code` / `auto` (default).
        /// Persisted to `.dora/config.toml` as `[source] mode = "..."`. If omitted, the mode
        /// already in the config file is preserved; if there's no config, auto-detect runs.
        #[arg(long)]
        mode: Option<String>,
    },
    /// Remove a source from the registry by name. Doesn't touch the source's `.dora/` dir.
    Remove {
        name: String,
    },
    /// List all registered sources.
    List,
    /// Set or clear the description for an existing source.
    Describe {
        name: String,
        /// New description text. Pass an empty string to clear.
        description: String,
    },
}

fn main() -> Result<()> {
    store::init_sqlite_vec();

    let cli = Cli::parse();
    match cli.command {
        Some(Command::Index { path, dry_run }) => {
            let source_root = path.unwrap_or_else(|| std::env::current_dir().expect("cwd"));
            cmd_index(&source_root, dry_run)
        }
        Some(Command::Mcp { source, include, exclude }) => cmd_mcp(source, include, exclude),
        Some(Command::Source { action }) => cmd_source(action),
        Some(Command::Install { client, include, exclude, no_shell, wrap }) => {
            cmd_install(client, include, exclude, !no_shell, wrap)
        }
        Some(Command::Doctor) => cmd_doctor(),
        Some(Command::Watch { include, exclude }) => cmd_watch(include, exclude),
        Some(Command::Wrappers { action }) => cmd_wrappers(action),
        Some(Command::Context { action }) => cmd_context(action),
        None => {
            let q = cli
                .query
                .context("provide a query, or use `dora index <path>` first")?;
            let cwd = std::env::current_dir()?;
            cmd_search(&cwd, &q, cli.top_k, cli.json, cli.min_score, cli.all, cli.files)
        }
    }
}

fn cmd_mcp(
    source_arg: Option<PathBuf>,
    include: Vec<String>,
    exclude: Vec<String>,
) -> Result<()> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("build tokio runtime for mcp")?;
    match source_arg {
        Some(path) => rt.block_on(mcp::run(&path)),
        None => {
            let mut reg = registry::Registry::load().context("load registry")?;
            if reg.sources.is_empty() {
                bail!(
                    "no sources registered. Add one with `dora source add <path>`, or run \
                     `dora mcp --source <path>` for a one-off."
                );
            }
            filter_registry(&mut reg, &include, &exclude)?;
            rt.block_on(mcp::run_multi(reg))
        }
    }
}

/// Apply --include / --exclude filters in-place. Validates each name against the registry
/// so typos fail fast with the list of valid names.
fn filter_registry(
    reg: &mut registry::Registry,
    include: &[String],
    exclude: &[String],
) -> Result<()> {
    if include.is_empty() && exclude.is_empty() {
        return Ok(());
    }
    let known: Vec<String> = reg.sources.iter().map(|s| s.name.clone()).collect();
    if !include.is_empty() {
        for name in include {
            if !known.contains(name) {
                bail!(
                    "no source named '{name}' in the registry. available: {}",
                    known.join(", ")
                );
            }
        }
        reg.sources.retain(|s| include.contains(&s.name));
    } else {
        for name in exclude {
            if !known.contains(name) {
                bail!(
                    "no source named '{name}' in the registry. available: {}",
                    known.join(", ")
                );
            }
        }
        reg.sources.retain(|s| !exclude.contains(&s.name));
    }
    if reg.sources.is_empty() {
        bail!("filter excluded every source — nothing left to serve. Check `dora source list`.");
    }
    Ok(())
}

fn cmd_install(
    client: Option<InstallClient>,
    include: Vec<String>,
    exclude: Vec<String>,
    do_shell: bool,
    wrap: Vec<String>,
) -> Result<()> {
    let target = match client {
        None => install::Client::All,
        Some(InstallClient::Claude) => install::Client::Claude,
        Some(InstallClient::Cursor) => install::Client::Cursor,
        Some(InstallClient::Codex) => install::Client::Codex,
    };
    let report = install::run(&include, &exclude, target, do_shell, &wrap)?;
    print!("{}", install::render_report(&report));
    Ok(())
}

fn cmd_wrappers(action: WrappersAction) -> Result<()> {
    use settings::Settings;
    match action {
        WrappersAction::On => {
            let mut s = Settings::load()?;
            s.wrappers.enabled = true;
            s.save()?;
            println!("dora wrappers: enabled");
            Ok(())
        }
        WrappersAction::Off => {
            let mut s = Settings::load()?;
            s.wrappers.enabled = false;
            s.save()?;
            println!(
                "dora wrappers: disabled — `grep`/`rg`/`ag`/`find` pass through to the real tool"
            );
            Ok(())
        }
        WrappersAction::Status { quiet } => {
            let s = Settings::load()?;
            if quiet {
                if s.wrappers.enabled {
                    std::process::exit(0);
                } else {
                    std::process::exit(1);
                }
            }
            let path = settings::settings_path()?;
            let path_note = if path.exists() {
                format!(" (config: {})", path.display())
            } else {
                String::new()
            };
            if s.wrappers.enabled {
                println!("dora wrappers: enabled{}", path_note);
            } else {
                println!(
                    "dora wrappers: disabled — `grep`/`rg`/`ag`/`find` pass through{}",
                    path_note
                );
            }
            Ok(())
        }
    }
}

fn cmd_context(action: ContextAction) -> Result<()> {
    fn open_store_for_source(name: &str) -> Result<Store> {
        let reg = registry::Registry::load().context("load registry")?;
        let src = reg
            .find_by_name(name)
            .ok_or_else(|| anyhow::anyhow!("no registered source named '{name}'"))?;
        let cfg = Config::load_or_default(&src.path).context("load config")?;
        let embedder = embed::from_config(&cfg.embedder, &models_dir(&src.path))?;
        let db = db_path(&src.path);
        if !db.exists() {
            bail!(
                ".dora/index.db not found at {}. Run `dora index {}` first.",
                src.path.display(),
                src.path.display()
            );
        }
        Store::open(&db, embedder.dims())
    }

    match action {
        ContextAction::Add { source, prefix, text } => {
            let store = open_store_for_source(&source)?;
            store.add_context(&prefix, &text)?;
            println!("context: {source} {prefix} → {text}");
            Ok(())
        }
        ContextAction::List { source } => {
            let store = open_store_for_source(&source)?;
            let rows = store.list_contexts()?;
            if rows.is_empty() {
                println!("(no contexts registered for '{source}')");
                return Ok(());
            }
            let prefix_w = rows.iter().map(|(p, _)| p.len()).max().unwrap_or(0).max(6);
            println!("{:<width$}  DESCRIPTION", "PREFIX", width = prefix_w);
            for (prefix, desc) in rows {
                println!("{:<width$}  {desc}", prefix, width = prefix_w);
            }
            Ok(())
        }
        ContextAction::Remove { source, prefix } => {
            let store = open_store_for_source(&source)?;
            let removed = store.remove_context(&prefix)?;
            if removed {
                println!("context removed: {source} {prefix}");
            } else {
                bail!("no context found at prefix '{prefix}' in source '{source}'");
            }
            Ok(())
        }
    }
}

fn cmd_watch(include: Vec<String>, exclude: Vec<String>) -> Result<()> {
    let mut reg = registry::Registry::load().context("load registry")?;
    // An empty registry isn't an error — the watcher waits and picks up the first
    // `dora source add` via its notify subscription on the registry directory. This is
    // load-bearing for `brew services start dora` on a fresh install.
    if !include.is_empty() || !exclude.is_empty() {
        filter_registry(&mut reg, &include, &exclude)?;
    }
    watch::run(reg.sources)
}

fn cmd_doctor() -> Result<()> {
    let report = doctor::run()?;
    print!("{}", doctor::render(&report));
    if report.errors() > 0 {
        std::process::exit(1);
    }
    Ok(())
}

fn cmd_source(action: SourceAction) -> Result<()> {
    match action {
        SourceAction::Add { path, name, description, mode } => {
            let abs = path
                .canonicalize()
                .with_context(|| format!("canonicalize {}", path.display()))?;
            // If --mode was given, write/replace `[source] mode = "..."` in the source's
            // config.toml *before* checking for the DB — so the user can do "add --mode code"
            // on an unindexed dir, then run `dora index` and have it pick up the right mode.
            if let Some(mode_str) = mode {
                let parsed = crate::mode::Mode::parse(&mode_str).ok_or_else(|| {
                    anyhow::anyhow!(
                        "invalid mode '{mode_str}'. valid: obsidian|notes|docs|code|auto"
                    )
                })?;
                write_source_mode(&abs, parsed.as_str())?;
                let resolved = crate::mode::Mode::resolve(&Some(mode_str.clone()), &abs);
                println!(
                    "mode: {} ({})",
                    resolved.as_str(),
                    crate::mode::detection_summary(&abs)
                );
            } else if !abs.join(".dora").join("config.toml").exists() {
                // No explicit mode, no existing config — auto-detect + print what we'd choose
                // so the user can confirm. Don't write the file: leaving config absent means
                // `dora index` will re-detect each run, which is friendlier for evolving dirs.
                let detected = crate::mode::Mode::detect(&abs);
                println!(
                    "mode: {} (auto-detected — {})",
                    detected.as_str(),
                    crate::mode::detection_summary(&abs)
                );
            }
            let db = abs.join(".dora").join("index.db");
            if !db.exists() {
                bail!(
                    "{} has no `.dora/index.db`. Run `dora index {}` first.",
                    abs.display(),
                    abs.display()
                );
            }
            let final_name = name.unwrap_or_else(|| {
                // Transcript-mode source roots have generic basenames (`projects`,
                // `sessions`) — use the mode name as a friendlier default.
                let resolved_mode = crate::mode::Mode::detect(&abs);
                match resolved_mode {
                    crate::mode::Mode::ClaudeCode => "claude-code".to_string(),
                    crate::mode::Mode::Codex => "codex".to_string(),
                    _ => abs
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_else(|| "source".to_string()),
                }
            });
            let mut reg = registry::Registry::load().context("load registry")?;
            reg.add(registry::Source {
                name: final_name.clone(),
                path: abs.clone(),
                description: description.filter(|s| !s.trim().is_empty()),
            })?;
            reg.save().context("write registry")?;
            println!("added: {} -> {}", final_name, abs.display());
            // Surface a one-line nudge about watch — uses kill -0 liveness so a stale
            // PID file from a crashed watcher doesn't give the wrong hint.
            if crate::watch::is_running() {
                eprintln!("hint: dora watch is running — it'll pick up '{final_name}' automatically.");
            } else {
                eprintln!(
                    "hint: run `dora watch` in the background to keep '{final_name}' indexed live."
                );
            }
            Ok(())
        }
        SourceAction::Remove { name } => {
            let mut reg = registry::Registry::load().context("load registry")?;
            let removed = reg.remove(&name)?;
            reg.save().context("write registry")?;
            println!("removed: {} ({})", removed.name, removed.path.display());
            Ok(())
        }
        SourceAction::List => {
            let reg = registry::Registry::load().context("load registry")?;
            if reg.sources.is_empty() {
                println!("(no sources registered — `dora source add <path>`)");
                return Ok(());
            }
            let name_w = reg.sources.iter().map(|s| s.name.len()).max().unwrap_or(4).max(4);
            println!("{:<width$}  STATUS  PATH", "NAME", width = name_w);
            for s in &reg.sources {
                let indexed = s.path.join(".dora").join("index.db").exists();
                let status = if indexed { "✓" } else { "✗" };
                println!(
                    "{:<width$}  {}      {}",
                    s.name,
                    status,
                    s.path.display(),
                    width = name_w,
                );
                if let Some(d) = &s.description {
                    println!("{:<width$}          {}", "", d, width = name_w);
                }
            }
            Ok(())
        }
        SourceAction::Describe { name, description } => {
            let mut reg = registry::Registry::load().context("load registry")?;
            let trimmed = if description.trim().is_empty() {
                None
            } else {
                Some(description)
            };
            reg.set_description(&name, trimmed.clone())?;
            reg.save().context("write registry")?;
            match trimmed {
                Some(d) => println!("{}: {}", name, d),
                None => println!("{}: (description cleared)", name),
            }
            Ok(())
        }
    }
}

fn dora_dir(vault: &Path) -> PathBuf {
    vault.join(".dora")
}
fn db_path(vault: &Path) -> PathBuf {
    dora_dir(vault).join("index.db")
}
fn models_dir(vault: &Path) -> PathBuf {
    dora_dir(vault).join("models")
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ---------------- index ----------------

fn cmd_index(vault: &Path, dry_run: bool) -> Result<()> {
    let started = Instant::now();
    let vault = vault.canonicalize().context("canonicalize vault path")?;
    std::fs::create_dir_all(dora_dir(&vault))?;

    let cfg = Config::load_or_default(&vault).context("load config")?;
    let embedder: DynEmbedder = embed::from_config(&cfg.embedder, &models_dir(&vault))?;
    let chunker: BoxedChunker = chunk::from_config(&cfg, &vault);

    // If the existing DB was built with a different schema/chunker/embedder, drop it and
    // rebuild from scratch (the diff loop will then see "all files = Insert"). No external
    // users → no migration code needed.
    let db = db_path(&vault);
    if db.exists() && !meta_matches(&db, embedder.as_ref())? {
        std::fs::remove_file(&db).context("remove stale index.db")?;
    }
    let mut store = Store::open(&db, embedder.dims())?;
    write_identity_meta(&store, embedder.as_ref())?;

    let summary = run_incremental_index(&vault, &cfg, &chunker, embedder.as_ref(), &mut store, dry_run)?;

    let settling_note = if summary.settling > 0 {
        format!(", {} settling", summary.settling)
    } else {
        String::new()
    };
    eprintln!(
        "{}: {} inserted, {} updated, {} touched, {} renamed, {} deleted, {} unchanged{} in {:.2?} [model: {}]",
        if dry_run { "dry-run" } else { "indexed" },
        summary.inserted,
        summary.updated,
        summary.touched,
        summary.renamed,
        summary.deleted,
        summary.skipped,
        settling_note,
        started.elapsed(),
        embedder.id(),
    );
    Ok(())
}

// ---------------- search ----------------

fn cmd_search(
    cwd: &Path,
    query: &str,
    top_k_override: Option<usize>,
    json: bool,
    min_score: Option<f64>,
    all: bool,
    files: bool,
) -> Result<()> {
    let source_root = cwd.canonicalize()?;
    let db = db_path(&source_root);
    if !db.exists() {
        bail!(
            ".dora/index.db not found in {}. Run `dora index` first.",
            source_root.display()
        );
    }

    let cfg = Config::load_or_default(&source_root).context("load config")?;
    let embedder: DynEmbedder = embed::from_config(&cfg.embedder, &models_dir(&source_root))?;
    let mut store = Store::open(&db, embedder.dims())?;
    check_meta(&store, embedder.as_ref())?;
    let chunker: BoxedChunker = chunk::from_config(&cfg, &source_root);

    // Derive source name: if cwd is registered, use the registered name; else use basename.
    let source_name = registry::find_source_name_for_path(&source_root).unwrap_or_else(|| {
        source_root
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "source".to_string())
    });

    let top_k = top_k_override.unwrap_or(cfg.search.top_k);
    let opts = search::SearchOptions {
        top_k,
        min_score,
        all,
        path_prefix: None,
        output: if files {
            search::OutputMode::Files
        } else {
            search::OutputMode::Chunks
        },
    };
    let hits = search_with_self_heal(
        &source_root,
        &source_name,
        &cfg,
        &mut store,
        &chunker,
        embedder.as_ref(),
        query,
        opts,
    )?;

    if json {
        println!("{}", serde_json::to_string_pretty(&hits)?);
        return Ok(());
    }
    if hits.is_empty() {
        eprintln!("no hits");
        return Ok(());
    }
    if files {
        // Files mode: one path per line, no decoration. Pairs cleanly with shell pipes.
        for h in hits {
            println!("{}", h.path);
        }
        return Ok(());
    }
    for h in hits {
        if h.heading_path.is_empty() {
            println!("{}:{}: {}", h.path, h.line, h.snippet);
        } else {
            println!("{}:{}: [{}] {}", h.path, h.line, h.heading_path, h.snippet);
        }
    }
    Ok(())
}

// ---------------- shared search helper ----------------

/// Fire a self-healing incremental walk if `last_walk_at` is stale (older than
/// `WALK_DEBOUNCE_SECS`), then run the hybrid search. Called from both `cmd_search` (CLI)
/// and the MCP `search` tool handler — single source of truth for the freshness story.
pub(crate) fn search_with_self_heal(
    source_root: &Path,
    source_name: &str,
    cfg: &Config,
    store: &mut Store,
    chunker: &dyn Chunker,
    embedder: &dyn Embedder,
    query: &str,
    opts: search::SearchOptions<'_>,
) -> Result<Vec<search::Hit>> {
    let last_walk: u64 = store
        .get_meta("last_walk_at")?
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    if now_secs().saturating_sub(last_walk) >= WALK_DEBOUNCE_SECS {
        run_incremental_index(source_root, cfg, chunker, embedder, store, false)?;
    }
    search::search(query, store, embedder, source_root, source_name, opts)
}

// ---------------- incremental indexing core ----------------

#[derive(Default)]
struct DiffSummary {
    inserted: usize,
    updated: usize,
    touched: usize,
    renamed: usize,
    deleted: usize,
    skipped: usize,
    /// Claude-code only: files filtered out by the settle window (active sessions).
    settling: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum WorkKind {
    Insert,
    Update,
}

/// One unit of "needs embedding" work. Held in memory between the diff phase and the execute
/// phase so the cost-preview can sum over actual to-be-embedded chunks (not total chunks).
struct EmbedWork {
    kind: WorkKind,
    path: String,
    mtime: u64,
    size: u64,
    content_hash: String,
    chunks: Vec<chunk::Chunk>,
    edges: Vec<chunk::EdgeSpec>,
    inputs: Vec<String>, // path/heading-prepended text fed to the embedder
}

fn run_incremental_index(
    vault: &Path,
    cfg: &Config,
    chunker: &dyn Chunker,
    embedder: &dyn Embedder,
    store: &mut Store,
    dry_run: bool,
) -> Result<DiffSummary> {
    // Phase 1: list entries (metadata only — no body reads). Extension allow-list comes
    // from the resolved mode so code sources walk .rs/.py/.ts/etc., not just .md.
    let mode = crate::mode::Mode::parse(&cfg.source.mode).unwrap_or(crate::mode::Mode::Notes);
    let allow_exts = mode.extensions();
    let mut entries = vault::list_entries(vault, &cfg.vault.ignore, allow_exts)?;

    // For claude-code sources, filter out files whose mtime is too recent — those are
    // active sessions being written to right now, and re-embedding every flush burns
    // the embedder for no benefit. They'll be picked up on a subsequent index pass
    // once they settle.
    let mut summary_settling: usize = 0;
    if mode.is_transcript() {
        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let cutoff = now_secs.saturating_sub(mode.settle_seconds(cfg));
        let before = entries.len();
        entries.retain(|e| e.mtime <= cutoff);
        summary_settling = before - entries.len();
    }

    // Phase 2: diff against existing files.
    let existing = store.list_files()?;
    let mut entry_paths: HashSet<String> = HashSet::with_capacity(entries.len());
    let mut to_insert: Vec<(String, String, u64, u64, String)> = Vec::new(); // (path, content, mtime, size, hash)
    let mut to_update: Vec<(String, String, u64, u64, String)> = Vec::new();
    let mut to_touch: Vec<(String, u64)> = Vec::new();
    let mut summary = DiffSummary::default();
    summary.settling = summary_settling;

    for entry in &entries {
        let rel = entry.relative_path.to_string_lossy().to_string();
        entry_paths.insert(rel.clone());

        match existing.get(&rel) {
            Some(row) if row.mtime == entry.mtime && row.size == entry.size => {
                summary.skipped += 1;
            }
            Some(row) => {
                // mtime or size changed → read body, hash, compare.
                let content = vault::read_file(vault, &entry.relative_path)?;
                let new_hash = sha256_hex(content.as_bytes());
                if new_hash == row.content_hash {
                    to_touch.push((rel, entry.mtime));
                } else {
                    to_update.push((rel, content, entry.mtime, entry.size, new_hash));
                }
            }
            None => {
                let content = vault::read_file(vault, &entry.relative_path)?;
                let new_hash = sha256_hex(content.as_bytes());
                to_insert.push((rel, content, entry.mtime, entry.size, new_hash));
            }
        }
    }

    // Phase 3: identify candidate deletes (in DB, not in vault).
    let mut to_delete: Vec<String> = existing
        .keys()
        .filter(|p| !entry_paths.contains(*p))
        .cloned()
        .collect();

    // Phase 4: rename detection — if an Insert shares a hash with a Delete, it's a rename.
    let mut delete_by_hash: HashMap<String, String> = HashMap::new();
    for path in &to_delete {
        if let Some(row) = existing.get(path) {
            delete_by_hash.insert(row.content_hash.clone(), path.clone());
        }
    }
    let mut to_rename: Vec<(String, String, u64, u64)> = Vec::new(); // (old_path, new_path, mtime, size)
    let mut absorbed: HashSet<String> = HashSet::new();
    let mut still_inserts: Vec<(String, String, u64, u64, String)> = Vec::new();
    for (new_path, content, mtime, size, hash) in to_insert {
        if let Some(old_path) = delete_by_hash.get(&hash) {
            to_rename.push((old_path.clone(), new_path, mtime, size));
            absorbed.insert(old_path.clone());
        } else {
            still_inserts.push((new_path, content, mtime, size, hash));
        }
    }
    to_delete.retain(|p| !absorbed.contains(p));

    // Phase 5: chunk all to-be-embedded files (held in memory for cost preview + execute).
    // Files that produce no chunks (empty, all-whitespace) are silently dropped — they
    // don't get a `files` row at all in this slice. Acceptable: they'll be re-read on every
    // run (small cost) but never embed. v0+ could persist them as zero-chunk marker rows.
    let mut work: Vec<EmbedWork> = Vec::new();
    let push_work = |work: &mut Vec<EmbedWork>,
                     kind: WorkKind,
                     path: &str,
                     content: &str,
                     mtime: u64,
                     size: u64,
                     hash: &str| {
        let rel_no_ext = Path::new(path)
            .with_extension("")
            .to_string_lossy()
            .to_string();
        let chunks = chunker.chunk(content, path);
        if chunks.is_empty() {
            return;
        }
        let edges = chunker.edges(content, path, &chunks);
        let inputs: Vec<String> = chunks
            .iter()
            .map(|c| chunk::embedded_text(&rel_no_ext, &c.heading_path, &c.content))
            .collect();
        work.push(EmbedWork {
            kind,
            path: path.to_string(),
            mtime,
            size,
            content_hash: hash.to_string(),
            chunks,
            edges,
            inputs,
        });
    };
    for (path, content, mtime, size, hash) in &still_inserts {
        push_work(&mut work, WorkKind::Insert, path, content, *mtime, *size, hash);
    }
    for (path, content, mtime, size, hash) in &to_update {
        push_work(&mut work, WorkKind::Update, path, content, *mtime, *size, hash);
    }

    // Cost preview (paid providers only): chunks that will actually be embedded.
    if let Some(per_million) = embedder.cost_per_million_tokens() {
        let total_chunks: usize = work.iter().map(|w| w.chunks.len()).sum();
        if total_chunks > 0 {
            let total_bytes: usize = work
                .iter()
                .flat_map(|w| w.inputs.iter())
                .map(|s| s.len())
                .sum();
            let est_tokens = total_bytes / 4;
            let est_cost = (est_tokens as f64 / 1_000_000.0) * per_million;
            eprintln!(
                "{} provider, {} chunks to embed (~{} tokens) → estimated cost ~${:.4}",
                embedder.id(),
                total_chunks,
                est_tokens,
                est_cost,
            );
        } else {
            eprintln!("{} provider, 0 chunks to embed (nothing changed)", embedder.id());
        }
    }

    if dry_run {
        // Report what *would* happen — counts work entries (excludes empty-chunk files).
        for w in &work {
            match w.kind {
                WorkKind::Insert => summary.inserted += 1,
                WorkKind::Update => summary.updated += 1,
            }
        }
        summary.touched = to_touch.len();
        summary.renamed = to_rename.len();
        summary.deleted = to_delete.len();
        return Ok(summary);
    }

    // Phase 6: execute non-embed actions first (cheap, atomic per row).
    for path in &to_delete {
        store.delete_file(path)?;
        summary.deleted += 1;
    }
    for (old, new, mtime, size) in &to_rename {
        store.rename_file(old, new, *mtime, *size)?;
        summary.renamed += 1;
    }
    for (path, mtime) in &to_touch {
        store.touch_file_mtime(path, *mtime)?;
        summary.touched += 1;
    }

    // Phase 7: embed + upsert per-file (per-file transaction inside Store::upsert).
    let mut any_edges = false;
    for w in &work {
        let embeddings = embedder.embed(&w.inputs)?;
        let rows: Vec<ChunkRow> = w
            .chunks
            .iter()
            .zip(embeddings.iter())
            .map(|(c, e)| ChunkRow {
                idx: c.idx,
                heading_path: &c.heading_path,
                content: &c.content,
                start_byte: c.start_byte,
                end_byte: c.end_byte,
                embedding: e,
                kind: chunk_kind_str(c.kind),
                symbol: c.symbol.as_deref(),
                parent_chunk_idx: c.parent_chunk_idx,
            })
            .collect();
        let link_rows: Vec<crate::store::LinkRow> = w
            .edges
            .iter()
            .map(|e| crate::store::LinkRow {
                source_chunk_idx: e.source_chunk_idx,
                kind: edge_kind_str(e.kind),
                target_symbol: &e.target_symbol,
                target_path: e.target_path.as_deref(),
            })
            .collect();
        if !link_rows.is_empty() {
            any_edges = true;
        }
        store.upsert_file_with_chunks(
            &w.path,
            w.mtime,
            w.size,
            &w.content_hash,
            &rows,
            &link_rows,
        )?;
        match w.kind {
            WorkKind::Insert => summary.inserted += 1,
            WorkKind::Update => summary.updated += 1,
        }
    }

    // Phase 7b: pass-2 cross-file edge resolution. Only meaningful for code sources, but
    // running it on a markdown-only source is a no-op (no links rows exist). Also runs
    // when files were deleted/renamed, since SET NULL may have orphaned links.
    if any_edges || !to_delete.is_empty() || !to_rename.is_empty() {
        let _resolved = store.resolve_cross_file_links()?;
    }

    // Phase 8: bump last_walk_at so cmd_search debounces.
    store.set_meta("last_walk_at", &now_secs().to_string())?;

    Ok(summary)
}

fn chunk_kind_str(k: chunk::ChunkKind) -> &'static str {
    use chunk::ChunkKind::*;
    match k {
        Prose => "prose",
        Function => "function",
        Method => "method",
        Class => "class",
        Struct => "struct",
        Trait => "trait",
        Interface => "interface",
        Impl => "impl",
        Enum => "enum",
        Module => "module",
        Const => "const",
        Macro => "macro",
    }
}

fn edge_kind_str(k: chunk::EdgeKind) -> &'static str {
    use chunk::EdgeKind::*;
    match k {
        Calls => "calls",
        References => "references",
        Implements => "implements",
        Imports => "imports",
        Extends => "extends",
    }
}

// ---------------- meta helpers ----------------

/// Quick read-only check whether the existing DB's identity matches the current binary + config.
/// Opens the connection without running `create_schema` so a stale DB with the previous schema
/// can be detected (and then wiped) without first tripping on a CREATE INDEX referencing a
/// column the old schema doesn't have.
fn meta_matches(db_path: &Path, embedder: &dyn Embedder) -> Result<bool> {
    let conn = rusqlite::Connection::open(db_path).context("open sqlite db (meta check)")?;
    let want = [
        ("schema_version", SCHEMA_VERSION.to_string()),
        ("chunker_version", CHUNKER_VERSION.to_string()),
        ("embedder_id", embedder.id().to_string()),
        ("embedder_dims", embedder.dims().to_string()),
    ];
    for (key, expected) in want {
        let got: Result<String, _> = conn.query_row(
            "SELECT value FROM meta WHERE key = ?",
            rusqlite::params![key],
            |r| r.get(0),
        );
        match got {
            Ok(v) if v == expected => continue,
            _ => return Ok(false),
        }
    }
    Ok(true)
}

fn write_identity_meta(store: &Store, embedder: &dyn Embedder) -> Result<()> {
    store.set_meta("schema_version", SCHEMA_VERSION)?;
    store.set_meta("chunker_version", CHUNKER_VERSION)?;
    store.set_meta("embedder_id", embedder.id())?;
    store.set_meta("embedder_dims", &embedder.dims().to_string())?;
    Ok(())
}

/// Refuse to search if the on-disk DB was built with different config/version than the current
/// binary or config.
fn check_meta(store: &Store, embedder: &dyn Embedder) -> Result<()> {
    let expected = [
        ("schema_version", SCHEMA_VERSION.to_string()),
        ("chunker_version", CHUNKER_VERSION.to_string()),
        ("embedder_id", embedder.id().to_string()),
        ("embedder_dims", embedder.dims().to_string()),
    ];
    for (key, want) in expected {
        match store.get_meta(key)? {
            Some(got) if got == want => continue,
            Some(got) => bail!(
                "index was built with {key}={got:?}, current setup expects {key}={want:?}. \
                 run `dora index` to rebuild."
            ),
            None => bail!(
                "index is missing {key} (likely built by an older dora). \
                 run `dora index` to rebuild."
            ),
        }
    }
    Ok(())
}

/// Write `[source] mode = "<value>"` into `<root>/.dora/config.toml`, preserving every
/// other line in the file. If the file doesn't exist yet, creates a minimal one. We do
/// line-level surgery rather than load → mutate → toml::to_string because the TOML library
/// drops comments + formatting on round-trip, which would be hostile to user-edited files.
fn write_source_mode(root: &Path, mode: &str) -> Result<()> {
    let dora = root.join(".dora");
    std::fs::create_dir_all(&dora).with_context(|| format!("mkdir {}", dora.display()))?;
    let path = dora.join("config.toml");
    let existing = if path.exists() {
        std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?
    } else {
        String::new()
    };

    let new_assignment = format!("mode = \"{mode}\"");
    let lines: Vec<&str> = existing.lines().collect();
    let mut out = String::new();
    let mut in_source_section = false;
    let mut wrote_mode = false;
    let mut has_source_section = false;

    for line in &lines {
        let trimmed = line.trim();
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            // Leaving the previous section. If we were in [source] and never wrote the mode,
            // append it now.
            if in_source_section && !wrote_mode {
                out.push_str(&new_assignment);
                out.push('\n');
                wrote_mode = true;
            }
            in_source_section = trimmed == "[source]";
            if in_source_section {
                has_source_section = true;
            }
            out.push_str(line);
            out.push('\n');
            continue;
        }
        if in_source_section && trimmed.starts_with("mode") && trimmed.contains('=') {
            out.push_str(&new_assignment);
            out.push('\n');
            wrote_mode = true;
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    if in_source_section && !wrote_mode {
        out.push_str(&new_assignment);
        out.push('\n');
        wrote_mode = true;
    }
    if !has_source_section {
        if !out.is_empty() && !out.ends_with("\n\n") {
            out.push('\n');
        }
        out.push_str("[source]\n");
        out.push_str(&new_assignment);
        out.push('\n');
    }
    std::fs::write(&path, out).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut s = String::with_capacity(digest.len() * 2);
    for b in digest {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

mod chunk;
mod config;
mod doctor;
mod embed;
mod install;
mod mcp;
mod registry;
mod search;
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
        None => {
            let q = cli
                .query
                .context("provide a query, or use `dora index <path>` first")?;
            let cwd = std::env::current_dir()?;
            cmd_search(&cwd, &q, cli.top_k, cli.json)
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

fn cmd_watch(include: Vec<String>, exclude: Vec<String>) -> Result<()> {
    let mut reg = registry::Registry::load().context("load registry")?;
    if reg.sources.is_empty() {
        bail!(
            "no sources registered. Add one with `dora source add <path>` before running `dora watch`."
        );
    }
    filter_registry(&mut reg, &include, &exclude)?;
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
        SourceAction::Add { path, name, description } => {
            let abs = path
                .canonicalize()
                .with_context(|| format!("canonicalize {}", path.display()))?;
            let db = abs.join(".dora").join("index.db");
            if !db.exists() {
                bail!(
                    "{} has no `.dora/index.db`. Run `dora index {}` first.",
                    abs.display(),
                    abs.display()
                );
            }
            let final_name = name.unwrap_or_else(|| {
                abs.file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| "source".to_string())
            });
            let mut reg = registry::Registry::load().context("load registry")?;
            reg.add(registry::Source {
                name: final_name.clone(),
                path: abs.clone(),
                description: description.filter(|s| !s.trim().is_empty()),
            })?;
            reg.save().context("write registry")?;
            println!("added: {} -> {}", final_name, abs.display());
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
    let chunker = Chunker::from_config(&cfg.chunking);

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

    eprintln!(
        "{}: {} inserted, {} updated, {} touched, {} renamed, {} deleted, {} unchanged in {:.2?} [model: {}]",
        if dry_run { "dry-run" } else { "indexed" },
        summary.inserted,
        summary.updated,
        summary.touched,
        summary.renamed,
        summary.deleted,
        summary.skipped,
        started.elapsed(),
        embedder.id(),
    );
    Ok(())
}

// ---------------- search ----------------

fn cmd_search(cwd: &Path, query: &str, top_k_override: Option<usize>, json: bool) -> Result<()> {
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
    let chunker = Chunker::from_config(&cfg.chunking);

    // Derive source name: if cwd is registered, use the registered name; else use basename.
    let source_name = registry::find_source_name_for_path(&source_root).unwrap_or_else(|| {
        source_root
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "source".to_string())
    });

    let top_k = top_k_override.unwrap_or(cfg.search.top_k);
    let hits = search_with_self_heal(
        &source_root,
        &source_name,
        &cfg,
        &mut store,
        &chunker,
        embedder.as_ref(),
        query,
        top_k,
        None,
    )?;

    if json {
        println!("{}", serde_json::to_string_pretty(&hits)?);
        return Ok(());
    }
    if hits.is_empty() {
        eprintln!("no hits");
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
    chunker: &Chunker,
    embedder: &dyn Embedder,
    query: &str,
    top_k: usize,
    path_prefix: Option<&str>,
) -> Result<Vec<search::Hit>> {
    let last_walk: u64 = store
        .get_meta("last_walk_at")?
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    if now_secs().saturating_sub(last_walk) >= WALK_DEBOUNCE_SECS {
        run_incremental_index(source_root, cfg, chunker, embedder, store, false)?;
    }
    search::search(query, store, embedder, source_root, source_name, top_k, path_prefix)
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
    inputs: Vec<String>, // path/heading-prepended text fed to the embedder
}

fn run_incremental_index(
    vault: &Path,
    cfg: &Config,
    chunker: &Chunker,
    embedder: &dyn Embedder,
    store: &mut Store,
    dry_run: bool,
) -> Result<DiffSummary> {
    // Phase 1: list entries (metadata only — no body reads).
    let entries = vault::list_entries(vault, &cfg.vault.ignore)?;

    // Phase 2: diff against existing files.
    let existing = store.list_files()?;
    let mut entry_paths: HashSet<String> = HashSet::with_capacity(entries.len());
    let mut to_insert: Vec<(String, String, u64, u64, String)> = Vec::new(); // (path, content, mtime, size, hash)
    let mut to_update: Vec<(String, String, u64, u64, String)> = Vec::new();
    let mut to_touch: Vec<(String, u64)> = Vec::new();
    let mut summary = DiffSummary::default();

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
        let chunks = chunker.chunk(content, &rel_no_ext);
        if chunks.is_empty() {
            return;
        }
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
            })
            .collect();
        store.upsert_file_with_chunks(&w.path, w.mtime, w.size, &w.content_hash, &rows)?;
        match w.kind {
            WorkKind::Insert => summary.inserted += 1,
            WorkKind::Update => summary.updated += 1,
        }
    }

    // Phase 8: bump last_walk_at so cmd_search debounces.
    store.set_meta("last_walk_at", &now_secs().to_string())?;

    Ok(summary)
}

// ---------------- meta helpers ----------------

/// Quick read-only check whether the existing DB's identity matches the current binary + config.
fn meta_matches(db_path: &Path, embedder: &dyn Embedder) -> Result<bool> {
    let store = Store::open(db_path, embedder.dims())?;
    let want = [
        ("schema_version", SCHEMA_VERSION.to_string()),
        ("chunker_version", CHUNKER_VERSION.to_string()),
        ("embedder_id", embedder.id().to_string()),
        ("embedder_dims", embedder.dims().to_string()),
    ];
    for (key, expected) in want {
        match store.get_meta(key)? {
            Some(got) if got == expected => continue,
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

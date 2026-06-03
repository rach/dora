mod chunk;
mod config;
mod doctor;
mod embed;
#[cfg(debug_assertions)]
mod eval;
mod graph;
mod install;
mod mcp;
mod migrations;
mod mode;
mod pagerank;
mod paths;
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
    about = "Local semantic memory for notes, code, and agent transcripts.",
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

    /// Include retrieval signals in JSON output. Normal pretty output ignores this.
    #[arg(long, global = true)]
    signals: bool,

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

    /// Repeatable. Intersect: a chunk must also score for this query. `dora "X" --and "Y"`
    /// ≈ "chunks about both X and Y". Each `--and` adds another hybrid search; the combined
    /// score is the harmonic mean of normalized per-query scores (asymmetry is punished).
    #[arg(long = "and", short = 'a', global = true, value_name = "QUERY")]
    and: Vec<String>,

    /// Repeatable. Exclude/demote: chunks scoring highly for this query are dropped;
    /// weaker matches are nudged down. `dora "X" --not "Z"` ≈ "X but not Z". Composes
    /// with `--and`.
    #[arg(long = "not", short = 'n', global = true, value_name = "QUERY")]
    not: Vec<String>,
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
        #[arg(
            long,
            value_name = "NAME",
            value_delimiter = ',',
            conflicts_with = "exclude"
        )]
        include: Vec<String>,

        /// Serve every registered source EXCEPT these.
        /// Mutually exclusive with --include and --source.
        #[arg(long, value_name = "NAME", value_delimiter = ',')]
        exclude: Vec<String>,

        /// Serve HTTP+JSON-RPC at `--bind:--port` (default `127.0.0.1:8181`) instead of stdio.
        /// Lets multiple MCP clients share one resident-models server. Pair with --daemon
        /// to fork into the background; without --daemon the server runs in the foreground.
        #[arg(long)]
        http: bool,

        /// HTTP bind address. Localhost-only by default; --bind 0.0.0.0 exposes beyond
        /// loopback and prints a warning at startup.
        #[arg(long, default_value = "127.0.0.1", value_name = "ADDR")]
        bind: String,

        /// HTTP port. Default 8181.
        #[arg(long, default_value_t = 8181, value_name = "N")]
        port: u16,

        /// Fork into the background; write PID to ~/.config/dora/mcp-http.pid.
        /// Requires --http. Unix only.
        #[arg(long)]
        daemon: bool,

        /// Disable the local read-only web UI normally served with --http.
        #[arg(long)]
        no_web: bool,

        /// Subcommand action — `stop` SIGTERMs a running daemon, `status` reports state.
        /// When absent and no --http flag, runs stdio (the v0–v0.4 default).
        #[command(subcommand)]
        action: Option<McpAction>,
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
    /// Migrate pre-0.9 co-located `<source>/.dora/` indexes into the centralized
    /// `~/.dora/sources/<name>/` layout. Sweeps every registered source; idempotent.
    Migrate,
    /// Foreground file-watcher that keeps registered sources fresh. Runs until Ctrl-C.
    /// Same `--include` / `--exclude` semantics as `dora mcp`.
    Watch {
        #[arg(
            long,
            value_name = "NAME",
            value_delimiter = ',',
            conflicts_with = "exclude"
        )]
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
    /// Show the wikilink graph for a note in the current source: which notes link to it
    /// (backlinks / inbound) and which it links to (outbound). Built at index time from
    /// `[[wikilinks]]` and `[text](note.md)` links.
    Backlinks {
        /// Note path relative to the source root, e.g. `Projects/dora.md`.
        path: String,
    },
    /// Document-graph maintenance.
    Graph {
        #[command(subcommand)]
        action: GraphAction,
    },
    /// Explain how dora ranked a query in the current source.
    Explain {
        /// Query to diagnose.
        query: String,
    },
    /// Run a committed retrieval eval fixture. Dev/debug builds only; absent from release.
    #[cfg(debug_assertions)]
    Eval {
        /// Fixture directory containing docs/ and queries.toml.
        fixture: PathBuf,
        /// Number of hits to evaluate per query. Default 5.
        #[arg(long, default_value_t = 5, value_name = "N")]
        top_k: usize,
        /// Optional minimum R@1 threshold. Exits nonzero if missed.
        #[arg(long, value_name = "FLOAT")]
        min_r_at_1: Option<f64>,
        /// Output JSON metrics + per-query outcomes.
        #[arg(long)]
        json: bool,
        /// Disable PRF for this eval run.
        #[arg(long)]
        disable_prf: bool,
        /// Disable graph boost for this eval run.
        #[arg(long)]
        disable_graph: bool,
        /// Run this fixture twice and require graph-on to beat graph-off on R@5 and MRR.
        #[arg(long)]
        compare_disable_graph: bool,
    },
}

#[derive(Subcommand)]
enum GraphAction {
    /// Force a full rebuild of the derived edges (keyphrase + similarity) for the current
    /// source. Normally runs automatically at index time; use this after tuning or to inspect.
    Rebuild,
}

#[derive(Subcommand)]
enum McpAction {
    /// Stop a running HTTP daemon. Reads `~/.config/dora/mcp-http.pid`, sends SIGTERM,
    /// waits up to 5s for the process to exit, then escalates to SIGKILL.
    Stop,
    /// Report whether the HTTP daemon is running + uptime + registered sources via GET
    /// `http://<bind>:<port>/health`. Exit 0 if running, 1 if not.
    Status,
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
    /// Register a source (or update its metadata if already registered). The folder does not
    /// need to be indexed first — register, then run `dora index`.
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
        /// Persisted to the source's central `config.toml` as `[source] mode = "..."`. If
        /// omitted, the mode already in the config file is preserved; if none, auto-detect runs.
        #[arg(long)]
        mode: Option<String>,
    },
    /// Remove a source from the registry by name. Keeps its index under `~/.dora/sources/<name>`
    /// unless `--purge` is given.
    Remove {
        name: String,
        /// Also delete the source's index data at `~/.dora/sources/<name>`.
        #[arg(long)]
        purge: bool,
    },
    /// Rename a registered source, moving its `~/.dora/sources/<name>` store to match.
    Rename {
        /// Current source name.
        old: String,
        /// New source name.
        new: String,
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
        Some(Command::Mcp {
            source,
            include,
            exclude,
            http,
            bind,
            port,
            daemon,
            no_web,
            action,
        }) => cmd_mcp(McpOptions {
            source,
            include,
            exclude,
            http,
            bind,
            port,
            daemon,
            web: !no_web,
            action,
        }),
        Some(Command::Source { action }) => cmd_source(action),
        Some(Command::Install {
            client,
            include,
            exclude,
            no_shell,
            wrap,
        }) => cmd_install(client, include, exclude, !no_shell, wrap),
        Some(Command::Doctor) => cmd_doctor(),
        Some(Command::Migrate) => cmd_migrate(),
        Some(Command::Watch { include, exclude }) => cmd_watch(include, exclude),
        Some(Command::Wrappers { action }) => cmd_wrappers(action),
        Some(Command::Context { action }) => cmd_context(action),
        Some(Command::Backlinks { path }) => {
            let cwd = std::env::current_dir()?;
            cmd_backlinks(&cwd, &path)
        }
        Some(Command::Graph { action }) => {
            let cwd = std::env::current_dir()?;
            cmd_graph(&cwd, action)
        }
        Some(Command::Explain { query }) => {
            let cwd = std::env::current_dir()?;
            cmd_explain(&cwd, &query, cli.top_k, cli.json)
        }
        #[cfg(debug_assertions)]
        Some(Command::Eval {
            fixture,
            top_k,
            min_r_at_1,
            json,
            disable_prf,
            disable_graph,
            compare_disable_graph,
        }) => eval::cmd_eval(
            &fixture,
            eval::EvalOptions {
                top_k,
                min_r_at_1,
                json,
                disable_prf,
                disable_graph,
                compare_disable_graph,
            },
        ),
        None => {
            let q = cli
                .query
                .context("provide a query, or use `dora index <path>` first")?;
            let cwd = std::env::current_dir()?;
            cmd_search(
                &cwd,
                &q,
                CliSearchOptions {
                    top_k_override: cli.top_k,
                    json: cli.json,
                    signals: cli.signals,
                    min_score: cli.min_score,
                    all: cli.all,
                    files: cli.files,
                    and_queries: cli.and,
                    not_queries: cli.not,
                },
            )
        }
    }
}

struct McpOptions {
    source: Option<PathBuf>,
    include: Vec<String>,
    exclude: Vec<String>,
    http: bool,
    bind: String,
    port: u16,
    daemon: bool,
    web: bool,
    action: Option<McpAction>,
}

fn cmd_mcp(opts: McpOptions) -> Result<()> {
    // Subcommand actions short-circuit: stop/status don't need to spin up a server.
    if let Some(action) = opts.action {
        return match action {
            McpAction::Stop => mcp_stop(&opts.bind, opts.port),
            McpAction::Status => mcp_status(&opts.bind, opts.port),
        };
    }

    // Optional fork-into-background. After this returns in the child, we continue normally;
    // the parent has already exited.
    if opts.daemon {
        if !opts.http {
            bail!("--daemon requires --http (stdio doesn't make sense as a background server)");
        }
        let pid_path = mcp_http_pid_path()?;
        if let Some(parent) = pid_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        if let Some(existing) = read_pid(&pid_path) {
            if process_alive(existing) {
                bail!("dora mcp http is already running on pid {existing}");
            } else {
                let _ = std::fs::remove_file(&pid_path);
            }
        }
        let log_path = pid_path.with_file_name("mcp-http.log");
        let stdout = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(&log_path)
            .with_context(|| format!("open daemon log {}", log_path.display()))?;
        let stderr = stdout.try_clone()?;
        daemonize::Daemonize::new()
            .pid_file(&pid_path)
            .working_directory(std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/")))
            .stdout(stdout)
            .stderr(stderr)
            .start()
            .context("daemonize")?;
        // Child continues below.
    }

    let rt = if opts.http {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .context("build tokio runtime for mcp http")?
    } else {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("build tokio runtime for mcp stdio")?
    };

    let transport = if opts.http {
        let addr: std::net::SocketAddr = format!("{}:{}", opts.bind, opts.port)
            .parse()
            .with_context(|| format!("invalid --bind/--port combo: {}:{}", opts.bind, opts.port))?;
        if !addr.ip().is_loopback() {
            eprintln!(
                "dora mcp: WARNING — exposing HTTP beyond loopback ({addr}). Indexed content \
                 is searchable by anyone reaching this address."
            );
        }
        mcp::Transport::Http {
            bind: addr,
            web: opts.web,
        }
    } else {
        mcp::Transport::Stdio
    };

    match opts.source {
        Some(path) if !opts.http => {
            // Ad-hoc single source: register it (storage is name-keyed) + migrate any legacy
            // co-located index, then serve a one-entry synthetic registry.
            let abs = path.canonicalize().context("canonicalize source path")?;
            let name = ensure_registered(&abs)?;
            migrate_source_if_legacy(&name, &abs)?;
            let reg = registry::Registry {
                sources: vec![registry::Source {
                    name,
                    path: abs,
                    description: None,
                }],
            };
            rt.block_on(mcp::run_multi(reg, transport))
        }
        Some(_) => {
            bail!("--source is incompatible with --http; HTTP daemon serves the full registry")
        }
        None => {
            let mut reg = registry::Registry::load().context("load registry")?;
            if reg.sources.is_empty() && !opts.http {
                bail!(
                    "no sources registered. Add one with `dora source add <path>`, or run \
                     `dora mcp --source <path>` for a one-off."
                );
            }
            filter_registry(&mut reg, &opts.include, &opts.exclude)?;
            rt.block_on(mcp::run_multi(reg, transport))
        }
    }
}

fn mcp_http_pid_path() -> Result<PathBuf> {
    let home = dirs::home_dir().context("could not determine $HOME")?;
    Ok(home.join(".config").join("dora").join("mcp-http.pid"))
}

fn read_pid(path: &Path) -> Option<u32> {
    std::fs::read_to_string(path)
        .ok()?
        .trim()
        .parse::<u32>()
        .ok()
}

fn process_alive(pid: u32) -> bool {
    std::process::Command::new("kill")
        .args(["-0", &pid.to_string()])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn mcp_stop(bind: &str, port: u16) -> Result<()> {
    let pid_path = mcp_http_pid_path()?;
    let Some(pid) = read_pid(&pid_path) else {
        eprintln!(
            "dora mcp: no daemon pid found at {} (not running?)",
            pid_path.display()
        );
        return Ok(());
    };
    if !process_alive(pid) {
        eprintln!("dora mcp: pid {pid} not alive — removing stale pid file");
        let _ = std::fs::remove_file(&pid_path);
        return Ok(());
    }
    eprintln!("dora mcp: SIGTERM pid {pid}");
    let _ = std::process::Command::new("kill")
        .args(["-TERM", &pid.to_string()])
        .status();
    // Wait up to 5s for graceful exit.
    for _ in 0..50 {
        if !process_alive(pid) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    if process_alive(pid) {
        eprintln!("dora mcp: escalating to SIGKILL");
        let _ = std::process::Command::new("kill")
            .args(["-KILL", &pid.to_string()])
            .status();
    }
    let _ = std::fs::remove_file(&pid_path);
    eprintln!("dora mcp: stopped (was on {bind}:{port})");
    Ok(())
}

fn mcp_status(bind: &str, port: u16) -> Result<()> {
    let url = format!("http://{bind}:{port}/health");
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_millis(500))
        .build()?;
    match client.get(&url).send().and_then(|r| r.text()) {
        Ok(body) => {
            println!("dora mcp: running");
            println!("  url:    {url}");
            println!("  health: {body}");
            Ok(())
        }
        Err(_) => {
            eprintln!("dora mcp: not running (no response at {url})");
            std::process::exit(1);
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
        let root = src.path.canonicalize().unwrap_or_else(|_| src.path.clone());
        migrate_source_if_legacy(name, &root)?;
        let cfg =
            Config::load_or_default(&root, &paths::config_path(name)?).context("load config")?;
        let embedder = embed::from_config(&cfg.embedder, &paths::models_root()?)?;
        let db = paths::db_path(name)?;
        if !db.exists() {
            bail!(
                "source '{name}' isn't indexed yet ({} missing). Run `dora index {}` first.",
                db.display(),
                root.display()
            );
        }
        Store::open(&db, embedder.dims())
    }

    match action {
        ContextAction::Add {
            source,
            prefix,
            text,
        } => {
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

fn cmd_migrate() -> Result<()> {
    let reg = registry::Registry::load().context("load registry")?;
    if reg.sources.is_empty() {
        println!("(no sources registered — nothing to migrate)");
        return Ok(());
    }
    let mut migrated = 0usize;
    let mut skipped = 0usize;
    for src in &reg.sources {
        let root = match src.path.canonicalize() {
            Ok(p) => p,
            Err(e) => {
                eprintln!("skip {} ({}): {e}", src.name, src.path.display());
                skipped += 1;
                continue;
            }
        };
        match migrate_source_if_legacy(&src.name, &root) {
            Ok(true) => migrated += 1,
            Ok(false) => skipped += 1,
            Err(e) => {
                eprintln!("migrate {} failed: {e}", src.name);
                skipped += 1;
            }
        }
    }
    // Regenerate the wrapper's source-roots file from the current registry.
    reg.write_roots_file().ok();
    println!("migrated {migrated}, skipped {skipped}");
    if migrated > 0 {
        println!("run `dora install` to refresh the shell wrappers for the new layout.");
    }
    Ok(())
}

fn cmd_source(action: SourceAction) -> Result<()> {
    match action {
        SourceAction::Add {
            path,
            name,
            description,
            mode,
        } => {
            let abs = path
                .canonicalize()
                .with_context(|| format!("canonicalize {}", path.display()))?;
            let desc = description.filter(|s| !s.trim().is_empty());
            let mut reg = registry::Registry::load().context("load registry")?;

            // Resolve the name first — central config + index are keyed by name. If the path is
            // already registered, treat `add` as a metadata update (rename is `source rename`).
            let final_name = if let Some(existing) = reg.find_by_path(&abs) {
                let nm = existing.name.clone();
                if desc.is_some() {
                    reg.set_description(&nm, desc.clone())?;
                    reg.save().context("write registry")?;
                }
                nm
            } else {
                let nm = match name {
                    Some(n) => {
                        registry::validate_source_name(&n)?;
                        n
                    }
                    None => {
                        let base = derive_source_name(&abs);
                        let mut nm = base.clone();
                        let mut k = 2;
                        while reg.find_by_name(&nm).is_some() {
                            nm = format!("{base}-{k}");
                            k += 1;
                        }
                        nm
                    }
                };
                reg.add(registry::Source {
                    name: nm.clone(),
                    path: abs.clone(),
                    description: desc.clone(),
                })?;
                reg.save().context("write registry")?;
                nm
            };

            migrate_source_if_legacy(&final_name, &abs)?;

            // If --mode was given, write/replace `[source] mode = "..."` in the source's central
            // config so a later `dora index` picks it up.
            if let Some(mode_str) = mode {
                let parsed = crate::mode::Mode::parse(&mode_str).ok_or_else(|| {
                    anyhow::anyhow!(
                        "invalid mode '{mode_str}'. valid: obsidian|notes|docs|code|auto"
                    )
                })?;
                write_source_mode(&paths::config_path(&final_name)?, parsed.as_str())?;
                let resolved = crate::mode::Mode::resolve(&Some(mode_str.clone()), &abs);
                println!(
                    "mode: {} ({})",
                    resolved.as_str(),
                    crate::mode::detection_summary(&abs)
                );
            } else if !paths::config_path(&final_name)?.exists() {
                let detected = crate::mode::Mode::detect(&abs);
                println!(
                    "mode: {} (auto-detected — {})",
                    detected.as_str(),
                    crate::mode::detection_summary(&abs)
                );
            }

            println!("added: {} -> {}", final_name, abs.display());
            if !paths::db_path(&final_name)?.exists() {
                eprintln!(
                    "hint: '{final_name}' isn't indexed yet — run `dora index {}`.",
                    abs.display()
                );
            }
            // Surface a one-line nudge about watch — uses kill -0 liveness so a stale
            // PID file from a crashed watcher doesn't give the wrong hint.
            if crate::watch::is_running() {
                eprintln!(
                    "hint: dora watch is running — it'll pick up '{final_name}' automatically."
                );
            } else {
                eprintln!(
                    "hint: run `dora watch` in the background to keep '{final_name}' indexed live."
                );
            }
            Ok(())
        }
        SourceAction::Remove { name, purge } => {
            let mut reg = registry::Registry::load().context("load registry")?;
            let removed = reg.remove(&name)?;
            reg.save().context("write registry")?;
            if purge {
                let dir = paths::source_store_dir(&removed.name)?;
                if dir.exists() {
                    std::fs::remove_dir_all(&dir)
                        .with_context(|| format!("remove {}", dir.display()))?;
                }
                println!(
                    "removed + purged: {} ({})",
                    removed.name,
                    removed.path.display()
                );
            } else {
                println!(
                    "removed: {} ({}) — index kept at {}",
                    removed.name,
                    removed.path.display(),
                    paths::source_store_dir(&removed.name)?.display()
                );
            }
            Ok(())
        }
        SourceAction::Rename { old, new } => {
            registry::validate_source_name(&new)?;
            let mut reg = registry::Registry::load().context("load registry")?;
            if reg.find_by_name(&new).is_some() {
                bail!("source name '{new}' already registered");
            }
            reg.find_by_name(&old)
                .ok_or_else(|| anyhow::anyhow!("no source named '{old}'"))?;
            // Move the central store dir to match the new name.
            let from = paths::source_store_dir(&old)?;
            let to = paths::source_store_dir(&new)?;
            if to.exists() {
                bail!("{} already exists", to.display());
            }
            if from.exists() {
                if let Some(parent) = to.parent() {
                    std::fs::create_dir_all(parent).ok();
                }
                std::fs::rename(&from, &to)
                    .with_context(|| format!("rename {} -> {}", from.display(), to.display()))?;
            }
            reg.rename(&old, &new)?;
            reg.save().context("write registry")?;
            println!("renamed: {old} -> {new}");
            Ok(())
        }
        SourceAction::List => {
            let reg = registry::Registry::load().context("load registry")?;
            if reg.sources.is_empty() {
                println!("(no sources registered — `dora source add <path>`)");
                return Ok(());
            }
            let name_w = reg
                .sources
                .iter()
                .map(|s| s.name.len())
                .max()
                .unwrap_or(4)
                .max(4);
            println!("{:<width$}  STATUS  PATH", "NAME", width = name_w);
            for s in &reg.sources {
                let indexed = paths::db_path(&s.name).map(|p| p.exists()).unwrap_or(false);
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
    let name = ensure_registered(&vault)?;
    migrate_source_if_legacy(&name, &vault)?;
    std::fs::create_dir_all(paths::source_store_dir(&name)?)?;

    let cfg =
        Config::load_or_default(&vault, &paths::config_path(&name)?).context("load config")?;
    let embedder: DynEmbedder = embed::from_config(&cfg.embedder, &paths::models_root()?)?;
    let chunker: BoxedChunker = chunk::from_config(&cfg, &vault);

    // If the existing DB was built with a different schema/chunker/embedder, drop it and
    // rebuild from scratch (the diff loop will then see "all files = Insert"). No external
    // users → no migration code needed.
    let db = paths::db_path(&name)?;
    if db.exists() && !meta_matches(&db, embedder.as_ref())? {
        std::fs::remove_file(&db).context("remove stale index.db")?;
    }
    let mut store = Store::open(&db, embedder.dims())?;
    write_identity_meta(&store, embedder.as_ref())?;

    let summary = run_incremental_index(
        &vault,
        &cfg,
        &chunker,
        embedder.as_ref(),
        &mut store,
        dry_run,
    )?;

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

struct CliSearchOptions {
    top_k_override: Option<usize>,
    json: bool,
    signals: bool,
    min_score: Option<f64>,
    all: bool,
    files: bool,
    and_queries: Vec<String>,
    not_queries: Vec<String>,
}

fn cmd_search(cwd: &Path, query: &str, cli_opts: CliSearchOptions) -> Result<()> {
    let (source_name, source_root) = resolve_source_for_cwd(cwd)?;
    migrate_source_if_legacy(&source_name, &source_root)?;
    let db = paths::db_path(&source_name)?;
    if !db.exists() {
        bail!(
            "source '{}' isn't indexed yet ({} missing). Run `dora index {}` first.",
            source_name,
            db.display(),
            source_root.display()
        );
    }

    let cfg = Config::load_or_default(&source_root, &paths::config_path(&source_name)?)
        .context("load config")?;
    let embedder: DynEmbedder = embed::from_config(&cfg.embedder, &paths::models_root()?)?;
    let mut store = Store::open(&db, embedder.dims())?;
    check_meta(&store, embedder.as_ref())?;
    let chunker: BoxedChunker = chunk::from_config(&cfg, &source_root);

    let top_k = cli_opts.top_k_override.unwrap_or(cfg.search.top_k);
    let opts = search::SearchOptions {
        top_k,
        min_score: cli_opts.min_score,
        all: cli_opts.all,
        path_prefix: None,
        output: if cli_opts.files {
            search::OutputMode::Files
        } else {
            search::OutputMode::Chunks
        },
        and_queries: cli_opts.and_queries,
        not_queries: cli_opts.not_queries,
        diagnostics: cli_opts.signals,
    };
    let hits = search_with_self_heal(
        SearchRuntime {
            source_root: &source_root,
            source_name: &source_name,
            cfg: &cfg,
            store: &mut store,
            chunker: &chunker,
            embedder: embedder.as_ref(),
        },
        query,
        opts,
    )?;

    if cli_opts.json {
        println!("{}", serde_json::to_string_pretty(&hits)?);
        return Ok(());
    }
    if hits.is_empty() {
        eprintln!("no hits");
        return Ok(());
    }
    if cli_opts.files {
        // Files mode: one path per line, no decoration. Pairs cleanly with shell pipes.
        for h in hits {
            println!("{}", h.path);
        }
        return Ok(());
    }
    let style = AnsiStyle::detect();
    for (i, h) in hits.iter().enumerate() {
        if i > 0 {
            println!();
        }
        render_hit_rich(&store, h, &style);
    }
    Ok(())
}

fn cmd_explain(cwd: &Path, query: &str, top_k_override: Option<usize>, json: bool) -> Result<()> {
    let (source_name, source_root) = resolve_source_for_cwd(cwd)?;
    migrate_source_if_legacy(&source_name, &source_root)?;
    let db = paths::db_path(&source_name)?;
    if !db.exists() {
        bail!(
            "source '{}' isn't indexed yet ({} missing). Run `dora index {}` first.",
            source_name,
            db.display(),
            source_root.display()
        );
    }

    let cfg = Config::load_or_default(&source_root, &paths::config_path(&source_name)?)
        .context("load config")?;
    let embedder: DynEmbedder = embed::from_config(&cfg.embedder, &paths::models_root()?)?;
    let mut store = Store::open(&db, embedder.dims())?;
    check_meta(&store, embedder.as_ref())?;
    let chunker: BoxedChunker = chunk::from_config(&cfg, &source_root);

    if now_secs().saturating_sub(
        store
            .get_meta("last_walk_at")?
            .and_then(|s| s.parse().ok())
            .unwrap_or(0),
    ) >= WALK_DEBOUNCE_SECS
    {
        run_incremental_index(
            &source_root,
            &cfg,
            &chunker,
            embedder.as_ref(),
            &mut store,
            false,
        )?;
    }

    let report = search::explain(
        query,
        &store,
        embedder.as_ref(),
        &source_root,
        &source_name,
        search::SearchOptions {
            top_k: top_k_override.unwrap_or(cfg.search.top_k),
            diagnostics: true,
            ..Default::default()
        },
    )?;

    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    println!("query: {}", report.query);
    println!("fts: {}", report.fts_query);
    if report.prf_terms.is_empty() {
        println!("prf: (none)");
    } else {
        println!("prf: {}", report.prf_terms.join(", "));
    }
    render_arm_summary("fts", &report.arms.fts);
    render_arm_summary("ann", &report.arms.ann);
    render_arm_summary("literal", &report.arms.literal);
    render_arm_summary("prf", &report.arms.prf);
    println!("\nfinal:");
    for h in &report.hits {
        let signals = h.signals.as_ref();
        println!(
            "  {:>7.4} {}:{} {}",
            h.score, h.path, h.line, h.heading_path
        );
        if let Some(s) = signals {
            println!(
                "          fts={:?} ann={:?} literal={:?} prf={:?} graph=+{:.4}",
                s.fts_rank, s.ann_rank, s.literal_rank, s.prf_rank, s.graph_boost
            );
        }
    }
    Ok(())
}

fn render_arm_summary(name: &str, hits: &[search::ArmHit]) {
    println!("\n{name}:");
    if hits.is_empty() {
        println!("  (none)");
        return;
    }
    for h in hits.iter().take(5) {
        println!("  #{:<2} {}:{} {}", h.rank, h.path, h.line, h.snippet);
    }
}

/// Minimalist `rg`-inspired text rendering: one header line per hit (path:line + heading
/// + score badge), then up to ~4 preview lines from the chunk indented under a thin bar.
///
/// Falls back to plain text + no decorations when stdout isn't a TTY (so pipes into jq,
/// grep, awk continue to see machine-parseable output).
fn render_hit_rich(store: &Store, h: &search::Hit, style: &AnsiStyle) {
    let header = format_header(h, style);
    println!("{header}");
    if let Some(ctx) = h.context.as_deref() {
        println!(
            "  {italic_dim}context: {ctx}{reset}",
            italic_dim = style.italic_dim,
            reset = style.reset,
        );
    }
    // Pull the chunk's full content for the preview. Best-effort: a missing chunk_id
    // (shouldn't happen — we just retrieved it) falls back to the precomputed snippet.
    let preview_text = match store.fetch_chunk(h.chunk_id) {
        Ok(Some(chunk)) => chunk.content,
        _ => h.snippet.clone(),
    };
    let preview_text = strip_leading_frontmatter_str(&preview_text);
    let mut emitted = 0usize;
    for line in preview_text.lines() {
        let trimmed = line.trim_end();
        // Drop blank and pure-heading-marker lines from the preview — they waste vertical
        // space without informing the reader. The first heading is already in the header
        // line we just printed.
        if trimmed.is_empty() || trimmed.starts_with("# ") || trimmed.starts_with("## ") {
            continue;
        }
        let truncated = truncate_chars(trimmed, 100);
        println!(
            "  {dim}\u{2502}{reset} {body}",
            dim = style.dim,
            reset = style.reset,
            body = truncated,
        );
        emitted += 1;
        if emitted >= 4 {
            break;
        }
    }
}

fn format_header(h: &search::Hit, style: &AnsiStyle) -> String {
    let path_line = format!(
        "{path_open}{path}{reset}:{line_open}{line}{reset}",
        path_open = style.path,
        path = h.path,
        reset = style.reset,
        line_open = style.line_num,
        line = h.line,
    );
    let heading = if h.heading_path.is_empty() {
        String::new()
    } else {
        format!(
            "  {hdr_open}[{hdr}]{reset}",
            hdr_open = style.heading,
            hdr = h.heading_path,
            reset = style.reset,
        )
    };
    let score = format!(
        "  {s_open}\u{2605}{score:.3}{reset}",
        s_open = style.score,
        score = h.score,
        reset = style.reset,
    );
    format!("{path_line}{heading}{score}")
}

fn truncate_chars(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let truncated: String = s.chars().take(max_chars).collect();
    format!("{truncated}\u{2026}")
}

/// Best-effort YAML-frontmatter stripper for preview text — mirrors `search::strip_leading_frontmatter`.
fn strip_leading_frontmatter_str(content: &str) -> &str {
    let trimmed = content.trim_start_matches('\u{feff}');
    let mut lines = trimmed.lines();
    let Some(first) = lines.next() else {
        return content;
    };
    if first.trim() != "---" {
        return content;
    }
    let mut consumed = first.len() + 1;
    for line in lines {
        consumed += line.len() + 1;
        if line.trim() == "---" {
            return &trimmed[consumed.min(trimmed.len())..];
        }
    }
    content
}

/// ANSI styling resolved once per invocation. Honors the `NO_COLOR` standard
/// (https://no-color.org/) and falls back to empty escape codes when stdout isn't a TTY,
/// so piping into `jq`, `awk`, or shell wrappers stays clean.
struct AnsiStyle {
    path: &'static str,
    line_num: &'static str,
    heading: &'static str,
    score: &'static str,
    dim: &'static str,
    italic_dim: &'static str,
    reset: &'static str,
}

impl AnsiStyle {
    fn detect() -> Self {
        use std::io::IsTerminal;
        let use_color = std::env::var_os("NO_COLOR").is_none() && std::io::stdout().is_terminal();
        if use_color {
            Self {
                path: "\x1b[1;35m",   // bold magenta
                line_num: "\x1b[32m", // green
                heading: "\x1b[36m",  // cyan
                score: "\x1b[2;33m",  // dim yellow
                dim: "\x1b[2m",
                italic_dim: "\x1b[2;3m",
                reset: "\x1b[0m",
            }
        } else {
            Self {
                path: "",
                line_num: "",
                heading: "",
                score: "",
                dim: "",
                italic_dim: "",
                reset: "",
            }
        }
    }
}

// ---------------- backlinks (Layer A graph) ----------------

fn cmd_backlinks(cwd: &Path, path: &str) -> Result<()> {
    let (source_name, source_root) = resolve_source_for_cwd(cwd)?;
    migrate_source_if_legacy(&source_name, &source_root)?;
    let db = paths::db_path(&source_name)?;
    if !db.exists() {
        bail!(
            "source '{}' isn't indexed yet ({} missing). Run `dora index {}` first.",
            source_name,
            db.display(),
            source_root.display()
        );
    }
    let cfg = Config::load_or_default(&source_root, &paths::config_path(&source_name)?)
        .context("load config")?;
    let embedder: DynEmbedder = embed::from_config(&cfg.embedder, &paths::models_root()?)?;
    let store = Store::open(&db, embedder.dims())?;

    let inbound = store.backlinks(path)?;
    let outbound = store.forward_links(path)?;

    if inbound.is_empty() && outbound.is_empty() {
        eprintln!("no wikilinks to or from {path}");
        return Ok(());
    }
    if !inbound.is_empty() {
        println!("{} note(s) link to {path}:", inbound.len());
        for p in &inbound {
            println!("  ← {p}");
        }
    }
    if !outbound.is_empty() {
        if !inbound.is_empty() {
            println!();
        }
        println!("{path} links to {} note(s):", outbound.len());
        for p in &outbound {
            println!("  → {p}");
        }
    }
    Ok(())
}

fn cmd_graph(cwd: &Path, action: GraphAction) -> Result<()> {
    match action {
        GraphAction::Rebuild => {
            let (source_name, source_root) = resolve_source_for_cwd(cwd)?;
            migrate_source_if_legacy(&source_name, &source_root)?;
            let db = paths::db_path(&source_name)?;
            if !db.exists() {
                bail!(
                    "source '{}' isn't indexed yet ({} missing). Run `dora index {}` first.",
                    source_name,
                    db.display(),
                    source_root.display()
                );
            }
            let cfg = Config::load_or_default(&source_root, &paths::config_path(&source_name)?)
                .context("load config")?;
            let embedder: DynEmbedder = embed::from_config(&cfg.embedder, &paths::models_root()?)?;
            let store = Store::open(&db, embedder.dims())?;
            let started = Instant::now();
            let n = graph::rebuild_derived_edges(&store, cfg.graph.entities)?;
            eprintln!("rebuilt {n} derived edge(s) in {:.2?}", started.elapsed());
            Ok(())
        }
    }
}

// ---------------- shared search helper ----------------

/// Fire a self-healing incremental walk if `last_walk_at` is stale (older than
/// `WALK_DEBOUNCE_SECS`), then run the hybrid search. Called from both `cmd_search` (CLI)
/// and the MCP `search` tool handler — single source of truth for the freshness story.
pub(crate) struct SearchRuntime<'a> {
    pub source_root: &'a Path,
    pub source_name: &'a str,
    pub cfg: &'a Config,
    pub store: &'a mut Store,
    pub chunker: &'a dyn Chunker,
    pub embedder: &'a dyn Embedder,
}

pub(crate) fn search_with_self_heal(
    rt: SearchRuntime<'_>,
    query: &str,
    opts: search::SearchOptions<'_>,
) -> Result<Vec<search::Hit>> {
    let last_walk: u64 = rt
        .store
        .get_meta("last_walk_at")?
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    if now_secs().saturating_sub(last_walk) >= WALK_DEBOUNCE_SECS {
        run_incremental_index(
            rt.source_root,
            rt.cfg,
            rt.chunker,
            rt.embedder,
            rt.store,
            false,
        )?;
    }
    search::search(
        query,
        rt.store,
        rt.embedder,
        rt.source_root,
        rt.source_name,
        opts,
    )
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

pub(crate) fn run_incremental_index(
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
    let mut summary = DiffSummary {
        settling: summary_settling,
        ..Default::default()
    };

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
            .map(|c| {
                let mut text = chunk::embedded_text(&rel_no_ext, &c.heading_path, &c.content);
                if let Some(symbol) = c.symbol.as_deref() {
                    let aliases = chunk::symbol_alias_text(&c.heading_path, symbol);
                    if !aliases.is_empty() {
                        text.push_str("\n\naliases:\n");
                        text.push_str(&aliases);
                    }
                }
                text
            })
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
        push_work(
            &mut work,
            WorkKind::Insert,
            path,
            content,
            *mtime,
            *size,
            hash,
        );
    }
    for (path, content, mtime, size, hash) in &to_update {
        push_work(
            &mut work,
            WorkKind::Update,
            path,
            content,
            *mtime,
            *size,
            hash,
        );
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
            eprintln!(
                "{} provider, 0 chunks to embed (nothing changed)",
                embedder.id()
            );
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

    // Phase 7b: pass-2 edge resolution. Code edges resolve by symbol; wikilinks resolve by
    // note title/path. Both no-op when their edge kind is absent. Also runs when files were
    // deleted/renamed, since SET NULL may have orphaned links.
    if any_edges || !to_delete.is_empty() || !to_rename.is_empty() {
        let _resolved = store.resolve_cross_file_links()?;
        let _wiki = store.resolve_wikilinks()?;
    }

    // Phase 8b: rebuild Layer-B derived edges (keyphrase + similarity) for prose sources when
    // the corpus changed. Code sources rely on the symbol graph instead. Derived edges are
    // global (kNN / co-occurrence reference all chunks), so any change triggers a full rebuild.
    let chunks_changed = summary.inserted > 0
        || summary.updated > 0
        || !to_delete.is_empty()
        || !to_rename.is_empty();
    if cfg.source.mode != "code" && chunks_changed {
        if let Err(e) = graph::rebuild_derived_edges(store, cfg.graph.entities) {
            eprintln!("dora: derived-edge rebuild failed (continuing): {e}");
        }
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
        Wikilink => "wikilink",
    }
}

// ---------------- meta helpers ----------------

/// Quick read-only check whether the existing DB's identity matches the current binary + config.
/// Opens the connection without running `create_schema` so a stale DB with the previous schema
/// can be detected (and then wiped) without first tripping on a CREATE INDEX referencing a
/// column the old schema doesn't have.
pub(crate) fn meta_matches(db_path: &Path, embedder: &dyn Embedder) -> Result<bool> {
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

pub(crate) fn write_identity_meta(store: &Store, embedder: &dyn Embedder) -> Result<()> {
    store.set_meta("schema_version", SCHEMA_VERSION)?;
    store.set_meta("chunker_version", CHUNKER_VERSION)?;
    store.set_meta("embedder_id", embedder.id())?;
    store.set_meta("embedder_dims", &embedder.dims().to_string())?;
    Ok(())
}

/// Refuse to search if the on-disk DB was built with different config/version than the current
/// binary or config.
pub(crate) fn check_meta(store: &Store, embedder: &dyn Embedder) -> Result<()> {
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

// ---------------- source identity, registration, migration ----------------

/// Friendly default name for a source root: its basename, except transcript dirs whose
/// basenames are generic (`projects`, `sessions`) get their mode name instead.
fn derive_source_name(root: &Path) -> String {
    match crate::mode::Mode::detect(root) {
        crate::mode::Mode::ClaudeCode => "claude-code".to_string(),
        crate::mode::Mode::Codex => "codex".to_string(),
        _ => root
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "source".to_string()),
    }
}

/// Ensure the canonicalized `root` is in the registry; return its name. With centralized,
/// name-keyed storage an index has nowhere to live until the source has a name, so `dora index`
/// (and ad-hoc `dora mcp --source`) auto-register rather than erroring. Reuses the existing
/// entry if the path is already registered; otherwise picks a basename-derived, collision-
/// deduped name.
fn ensure_registered(root: &Path) -> Result<String> {
    let mut reg = registry::Registry::load().context("load registry")?;
    if let Some(s) = reg.find_by_path(root) {
        return Ok(s.name.clone());
    }
    let base = derive_source_name(root);
    let mut name = base.clone();
    let mut n = 2;
    while reg.find_by_name(&name).is_some() {
        name = format!("{base}-{n}");
        n += 1;
    }
    reg.add(registry::Source {
        name: name.clone(),
        path: root.to_path_buf(),
        description: None,
    })?;
    reg.save().context("write registry")?;
    eprintln!("registered: {} -> {}", name, root.display());
    Ok(name)
}

/// Resolve the source that owns `cwd` for read commands (search/explain/backlinks/graph).
/// Order: (1) registry longest-prefix match; (2) legacy fallback — if an un-migrated
/// `<ancestor>/.dora/index.db` exists, auto-register that ancestor for a smooth pre-0.9
/// upgrade; (3) error. Returns `(name, canonical source root)`.
fn resolve_source_for_cwd(cwd: &Path) -> Result<(String, PathBuf)> {
    let cwd = cwd.canonicalize().context("canonicalize cwd")?;
    let reg = registry::Registry::load().context("load registry")?;
    if let Some(s) = reg.resolve_for_path(&cwd) {
        return Ok((s.name.clone(), s.path.clone()));
    }
    let mut dir = cwd.as_path();
    loop {
        if paths::legacy_dir(dir).join("index.db").exists() {
            let root = dir.to_path_buf();
            let name = ensure_registered(&root)?;
            return Ok((name, root));
        }
        match dir.parent() {
            Some(p) => dir = p,
            None => break,
        }
    }
    bail!(
        "no registered dora source contains {}. Run `dora index <path>` to index + register it.",
        cwd.display()
    );
}

/// Force a SQLite WAL checkpoint so `index.db` is self-contained before it's moved.
fn checkpoint_wal(db: &Path) -> Result<()> {
    let conn = rusqlite::Connection::open(db)?;
    conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")?;
    Ok(())
}

/// Move a file or directory, falling back to copy+remove when `rename` fails across mounts
/// (EXDEV — `~/.dora` and the source may be on different volumes). The source is removed only
/// after the destination is in place.
fn move_path(from: &Path, to: &Path) -> Result<()> {
    if std::fs::rename(from, to).is_ok() {
        return Ok(());
    }
    if from.is_dir() {
        copy_dir_recursive(from, to)?;
        std::fs::remove_dir_all(from)?;
    } else {
        std::fs::copy(from, to)
            .with_context(|| format!("copy {} -> {}", from.display(), to.display()))?;
        std::fs::remove_file(from)?;
    }
    Ok(())
}

fn copy_dir_recursive(from: &Path, to: &Path) -> Result<()> {
    std::fs::create_dir_all(to)?;
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        let dest = to.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir_recursive(&entry.path(), &dest)?;
        } else {
            std::fs::copy(entry.path(), &dest)?;
        }
    }
    Ok(())
}

/// Move a pre-0.9 co-located `<root>/.dora` store into the centralized location keyed by
/// `name`. Idempotent and crash-safe; returns `Ok(false)` when there's nothing to migrate.
/// Called at the top of every command that opens a source so existing installs upgrade on
/// first touch with no user action.
fn migrate_source_if_legacy(name: &str, root: &Path) -> Result<bool> {
    let legacy = paths::legacy_dir(root);
    let legacy_db = legacy.join("index.db");
    if !legacy_db.exists() {
        return Ok(false); // already migrated, or never co-located
    }
    let central_dir = paths::source_store_dir(name)?;
    let central_db = paths::db_path(name)?;

    // Both present → the central copy wins (a fresh `dora index` rebuilds cheaply). Drop legacy.
    if central_db.exists() {
        std::fs::remove_dir_all(&legacy)
            .with_context(|| format!("remove stale legacy {}", legacy.display()))?;
        return Ok(true);
    }

    std::fs::create_dir_all(&central_dir)
        .with_context(|| format!("create {}", central_dir.display()))?;

    // Checkpoint the WAL so all data is in index.db, then move db + sidecars.
    checkpoint_wal(&legacy_db).ok();
    for suffix in ["", "-wal", "-shm"] {
        let from = legacy.join(format!("index.db{suffix}"));
        if from.exists() {
            move_path(&from, &central_dir.join(format!("index.db{suffix}")))?;
        }
    }
    // Move the per-source config.
    let legacy_cfg = legacy.join("config.toml");
    if legacy_cfg.exists() {
        move_path(&legacy_cfg, &paths::config_path(name)?)?;
    }
    // Move model subtrees into the shared cache; drop any that are already cached (dedup).
    let legacy_models = legacy.join("models");
    if legacy_models.is_dir() {
        let shared = paths::models_root()?;
        std::fs::create_dir_all(&shared).ok();
        for entry in std::fs::read_dir(&legacy_models)? {
            let entry = entry?;
            let dest = shared.join(entry.file_name());
            if dest.exists() {
                if entry.path().is_dir() {
                    std::fs::remove_dir_all(entry.path()).ok();
                } else {
                    std::fs::remove_file(entry.path()).ok();
                }
            } else {
                move_path(&entry.path(), &dest)?;
            }
        }
    }
    // Everything moved — drop the now-empty legacy dir, leaving zero footprint in the folder.
    std::fs::remove_dir_all(&legacy)
        .with_context(|| format!("remove migrated legacy {}", legacy.display()))?;
    eprintln!(
        "migrated {name}: moved index out of {} → {}",
        legacy.display(),
        central_dir.display()
    );
    Ok(true)
}

// ---------------- config writer ----------------

/// Write `[source] mode = "<value>"` into the source's central `config.toml`, preserving every
/// other line in the file. If the file doesn't exist yet, creates a minimal one. We do
/// line-level surgery rather than load → mutate → toml::to_string because the TOML library
/// drops comments + formatting on round-trip, which would be hostile to user-edited files.
fn write_source_mode(config_path: &Path, mode: &str) -> Result<()> {
    if let Some(parent) = config_path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("mkdir {}", parent.display()))?;
    }
    let path = config_path;
    let existing = if path.exists() {
        std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?
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
    }
    if !has_source_section {
        if !out.is_empty() && !out.ends_with("\n\n") {
            out.push('\n');
        }
        out.push_str("[source]\n");
        out.push_str(&new_assignment);
        out.push('\n');
    }
    std::fs::write(path, out).with_context(|| format!("write {}", path.display()))?;
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

#[cfg(test)]
mod migrate_tests {
    use std::sync::Mutex;

    // `DORA_HOME` is process-global; serialize env-touching tests.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn migrates_legacy_store_into_central() {
        let _g = ENV_LOCK.lock().unwrap();
        let home = tempfile::tempdir().unwrap();
        std::env::set_var("DORA_HOME", home.path());

        // A fake source folder carrying a pre-0.9 co-located `.dora/`.
        let srcdir = tempfile::tempdir().unwrap();
        let legacy = srcdir.path().join(".dora");
        let model_subtree = legacy.join("models").join("models--acme--m1");
        std::fs::create_dir_all(&model_subtree).unwrap();
        std::fs::write(legacy.join("index.db"), b"fake-sqlite").unwrap();
        std::fs::write(legacy.join("config.toml"), "[source]\nmode = \"notes\"\n").unwrap();
        std::fs::write(model_subtree.join("w.bin"), b"weights").unwrap();

        let moved = super::migrate_source_if_legacy("testsrc", srcdir.path()).unwrap();
        assert!(moved);

        // Central store now holds the db + config; the folder is left with zero footprint.
        let central = crate::paths::source_store_dir("testsrc").unwrap();
        assert!(central.join("index.db").exists());
        assert!(central.join("config.toml").exists());
        assert!(!srcdir.path().join(".dora").exists());

        // Model subtree landed in the shared cache.
        assert!(crate::paths::models_root()
            .unwrap()
            .join("models--acme--m1")
            .join("w.bin")
            .exists());

        // Idempotent: nothing left to migrate on a second pass.
        assert!(!super::migrate_source_if_legacy("testsrc", srcdir.path()).unwrap());

        std::env::remove_var("DORA_HOME");
    }
}

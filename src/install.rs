//! `dora install` — auto-patch MCP host configs so users don't hand-edit JSON. Also injects
//! the zsh `grep` wrapper into `~/.zshrc` (per SPEC). Idempotent: re-running re-applies the
//! same `dora` block in place without duplicating or disturbing other entries.

use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

/// What `dora install` did for one host (Claude/Cursor/Codex).
#[derive(Debug)]
pub enum HostAction {
    /// Config file didn't exist — client probably not installed. Skipped.
    NotInstalled,
    /// Skipped because `--client <other>` filtered this host out.
    SkippedByFilter,
    /// Already had the same dora block. No-op.
    AlreadyUpToDate,
    /// Patched the file (either added a new dora block or updated an existing one).
    Patched,
    /// Tried to patch but ran into a problem worth surfacing.
    Failed(String),
}

#[derive(Debug)]
pub struct InstallReport {
    pub binary_path: PathBuf,
    pub args: Vec<String>,
    pub claude: (PathBuf, HostAction),
    pub cursor: (PathBuf, HostAction),
    pub codex: (PathBuf, HostAction),
    pub shell: ShellAction,
}

#[derive(Debug)]
pub enum ShellAction {
    /// Per-tool result: `(tool, what happened)`. Empty vec → no zsh ops attempted.
    Done(Vec<(String, WrapperOp)>),
    Skipped(String),
}

#[derive(Debug)]
pub enum WrapperOp {
    /// Block didn't exist; appended a fresh one.
    Added,
    /// Block existed but content drifted; rewrote it.
    Updated,
    /// Block existed with exactly the desired content; no write.
    AlreadyUpToDate,
    /// Block existed but wasn't in the requested set; removed it.
    Removed,
    /// Tool name not recognized.
    UnknownTool,
}

/// List of supported wrapper tool names.
pub const SUPPORTED_WRAPS: &[&str] = &["grep", "rg", "ag", "find"];

/// Targeted single client for `dora install --client <name>`.
#[derive(Debug, Clone, Copy)]
pub enum Client {
    All,
    Claude,
    Cursor,
    Codex,
}

pub fn run(
    include: &[String],
    exclude: &[String],
    client: Client,
    do_shell: bool,
    wraps: &[String],
) -> Result<InstallReport> {
    let binary_path = std::env::current_exe().context("locate current dora binary")?;
    let args = build_mcp_args(include, exclude);

    let home = dirs::home_dir().context("could not determine $HOME")?;

    let claude_path = home.join(".claude.json");
    let cursor_path = home.join(".cursor").join("mcp.json");
    let codex_path = home.join(".codex").join("config.toml");

    let claude = if matches!(client, Client::All | Client::Claude) {
        patch_json_host(&claude_path, &binary_path, &args)
    } else {
        HostAction::SkippedByFilter
    };
    let cursor = if matches!(client, Client::All | Client::Cursor) {
        patch_json_host(&cursor_path, &binary_path, &args)
    } else {
        HostAction::SkippedByFilter
    };
    let codex = if matches!(client, Client::All | Client::Codex) {
        patch_toml_host(&codex_path, &binary_path, &args)
    } else {
        HostAction::SkippedByFilter
    };

    let shell = if do_shell {
        inject_zsh_wrappers(&home, wraps)
    } else {
        ShellAction::Skipped("--no-shell flag".to_string())
    };

    Ok(InstallReport {
        binary_path,
        args,
        claude: (claude_path, claude),
        cursor: (cursor_path, cursor),
        codex: (codex_path, codex),
        shell,
    })
}

fn build_mcp_args(include: &[String], exclude: &[String]) -> Vec<String> {
    let mut args = vec!["mcp".to_string()];
    if !include.is_empty() {
        args.push("--include".to_string());
        args.push(include.join(","));
    } else if !exclude.is_empty() {
        args.push("--exclude".to_string());
        args.push(exclude.join(","));
    }
    args
}

// ---------- JSON-format hosts (Claude Code, Cursor) ----------

fn patch_json_host(path: &Path, binary_path: &Path, args: &[String]) -> HostAction {
    if !path.exists() {
        return HostAction::NotInstalled;
    }

    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => return HostAction::Failed(format!("read: {e}")),
    };

    let mut root: Value = if text.trim().is_empty() {
        json!({})
    } else {
        match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(e) => return HostAction::Failed(format!("parse json: {e}")),
        }
    };

    if !root.is_object() {
        return HostAction::Failed("root is not an object".to_string());
    }

    let target = json!({
        "command": binary_path.display().to_string(),
        "args": args,
    });

    // Ensure mcpServers exists as an object.
    let mcp_servers = root
        .as_object_mut()
        .unwrap()
        .entry("mcpServers")
        .or_insert_with(|| json!({}));
    if !mcp_servers.is_object() {
        return HostAction::Failed("mcpServers is not an object".to_string());
    }
    let mcp_map = mcp_servers.as_object_mut().unwrap();

    if let Some(existing) = mcp_map.get("dora") {
        if existing == &target {
            return HostAction::AlreadyUpToDate;
        }
    }

    mcp_map.insert("dora".to_string(), target);

    let pretty = match serde_json::to_string_pretty(&root) {
        Ok(s) => s,
        Err(e) => return HostAction::Failed(format!("serialize: {e}")),
    };
    if let Err(e) = atomic_write(path, pretty.as_bytes()) {
        return HostAction::Failed(format!("write: {e}"));
    }
    HostAction::Patched
}

// ---------- TOML-format hosts (Codex CLI) ----------

fn patch_toml_host(path: &Path, binary_path: &Path, args: &[String]) -> HostAction {
    if !path.exists() {
        return HostAction::NotInstalled;
    }

    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => return HostAction::Failed(format!("read: {e}")),
    };

    let mut root: toml::Value = match toml::from_str(&text) {
        Ok(v) => v,
        Err(e) => return HostAction::Failed(format!("parse toml: {e}")),
    };

    // Codex MCP convention: [mcp_servers.dora] command = "..." args = [...]
    let table = match root.as_table_mut() {
        Some(t) => t,
        None => return HostAction::Failed("root is not a table".to_string()),
    };

    let mcp_servers = table
        .entry("mcp_servers".to_string())
        .or_insert_with(|| toml::Value::Table(toml::value::Table::new()));
    let mcp_map = match mcp_servers.as_table_mut() {
        Some(t) => t,
        None => return HostAction::Failed("mcp_servers is not a table".to_string()),
    };

    let mut new_entry = toml::value::Table::new();
    new_entry.insert(
        "command".to_string(),
        toml::Value::String(binary_path.display().to_string()),
    );
    new_entry.insert(
        "args".to_string(),
        toml::Value::Array(
            args.iter()
                .map(|s| toml::Value::String(s.clone()))
                .collect(),
        ),
    );
    let target = toml::Value::Table(new_entry);

    if let Some(existing) = mcp_map.get("dora") {
        if existing == &target {
            return HostAction::AlreadyUpToDate;
        }
    }

    mcp_map.insert("dora".to_string(), target);

    let serialized = match toml::to_string_pretty(&root) {
        Ok(s) => s,
        Err(e) => return HostAction::Failed(format!("serialize: {e}")),
    };
    if let Err(e) = atomic_write(path, serialized.as_bytes()) {
        return HostAction::Failed(format!("write: {e}"));
    }
    HostAction::Patched
}

// ---------- zsh tool wrappers ----------

fn marker_begin(tool: &str) -> String {
    format!("# >>> dora {tool} wrapper >>>")
}
fn marker_end(tool: &str) -> String {
    format!("# <<< dora {tool} wrapper <<<")
}

/// Body for a `grep`/`rg`/`ag`-style wrapper: any flag falls through, otherwise route to dora
/// when inside a registered source.
fn standard_body(tool: &str) -> String {
    let begin = marker_begin(tool);
    let end = marker_end(tool);
    format!(
        r#"{begin}
# dora {tool} wrapper — flagless `{tool}` inside a registered dora source routes to semantic
# search. Any flag (-i, -F, etc.) or non-source cwd → real {tool}, unchanged.
{tool}() {{
    for arg in "$@"; do
        case "$arg" in -*) command {tool} "$@"; return ;;
        esac
    done
    _dora_dir="$PWD"
    while [ "$_dora_dir" != "/" ]; do
        if [ -f "$_dora_dir/.dora/index.db" ]; then
            dora "$@"
            return
        fi
        _dora_dir="$(dirname "$_dora_dir")"
    done
    command {tool} "$@"
}}
{end}
"#
    )
}

/// `find` needs a stricter heuristic — it's almost always invoked with flags or paths.
/// We only intercept the *single quoted phrase* form: `find "rust lifetimes"`. Any other
/// shape falls through to real find unchanged.
fn find_body() -> String {
    let begin = marker_begin("find");
    let end = marker_end("find");
    format!(
        r#"{begin}
# dora find wrapper — `find "natural phrase"` (exactly one arg containing whitespace, no flags)
# inside a registered dora source routes to semantic search. Any other invocation shape
# (with flags, with paths, with multiple args) → real find, unchanged.
find() {{
    if [ "$#" -eq 1 ]; then
        case "$1" in
            -*) ;;
            *' '*)
                _dora_dir="$PWD"
                while [ "$_dora_dir" != "/" ]; do
                    if [ -f "$_dora_dir/.dora/index.db" ]; then
                        dora "$1"
                        return
                    fi
                    _dora_dir="$(dirname "$_dora_dir")"
                done
                ;;
        esac
    fi
    command find "$@"
}}
{end}
"#
    )
}

fn wrapper_body(tool: &str) -> Option<String> {
    match tool {
        "grep" | "rg" | "ag" => Some(standard_body(tool)),
        "find" => Some(find_body()),
        _ => None,
    }
}

/// Inject the requested wrappers, removing any previously-installed wrappers NOT in the set.
/// `wraps` is authoritative — re-running with `--wrap rg` after `--wrap grep,rg` removes grep.
fn inject_zsh_wrappers(home: &Path, wraps: &[String]) -> ShellAction {
    let zshrc = home.join(".zshrc");
    let mut text = match std::fs::read_to_string(&zshrc) {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => return ShellAction::Skipped(format!("read .zshrc: {e}")),
    };

    let mut ops: Vec<(String, WrapperOp)> = Vec::new();
    let requested: std::collections::HashSet<&str> = wraps.iter().map(|s| s.as_str()).collect();

    // Step 1: remove blocks for any installed-but-not-requested tool. Also blocks for
    // unknown tools (e.g., from earlier dora versions or hand-edits) get left alone.
    for tool in SUPPORTED_WRAPS {
        if requested.contains(*tool) {
            continue;
        }
        if let Some((start, end)) = block_span(&text, tool) {
            text = strip_block(&text, start, end);
            ops.push(((*tool).to_string(), WrapperOp::Removed));
        }
    }

    // Step 2: for each requested tool, add/update/no-op as needed.
    for tool in wraps {
        let body = match wrapper_body(tool) {
            Some(b) => b,
            None => {
                ops.push((tool.clone(), WrapperOp::UnknownTool));
                continue;
            }
        };
        let body_trimmed = body.trim_end_matches('\n').to_string();
        match block_span(&text, tool) {
            Some((start, end)) => {
                if &text[start..end] == body_trimmed.as_str() {
                    ops.push((tool.clone(), WrapperOp::AlreadyUpToDate));
                } else {
                    let mut new = String::with_capacity(text.len());
                    new.push_str(&text[..start]);
                    new.push_str(&body_trimmed);
                    new.push_str(&text[end..]);
                    text = new;
                    ops.push((tool.clone(), WrapperOp::Updated));
                }
            }
            None => {
                if !text.is_empty() && !text.ends_with('\n') {
                    text.push('\n');
                }
                text.push_str(&body);
                ops.push((tool.clone(), WrapperOp::Added));
            }
        }
    }

    // Step 3: only write if something actually changed.
    let need_write = ops.iter().any(|(_, op)| {
        !matches!(op, WrapperOp::AlreadyUpToDate | WrapperOp::UnknownTool)
    });
    if need_write {
        if let Err(e) = atomic_write(&zshrc, text.as_bytes()) {
            return ShellAction::Skipped(format!("write .zshrc: {e}"));
        }
    }
    ShellAction::Done(ops)
}

/// Find `(begin_marker_index, end_after_end_marker)` for the dora wrapper block of `tool`,
/// or None if absent.
fn block_span(text: &str, tool: &str) -> Option<(usize, usize)> {
    let begin = marker_begin(tool);
    let end = marker_end(tool);
    let start = text.find(&begin)?;
    let after_start = start + begin.len();
    let rel_end = text[after_start..].find(&end)?;
    Some((start, after_start + rel_end + end.len()))
}

/// Remove `text[start..end]` plus a single trailing newline if present (keeps the file tidy).
fn strip_block(text: &str, start: usize, end: usize) -> String {
    let trailing_nl = if text.as_bytes().get(end) == Some(&b'\n') { 1 } else { 0 };
    let mut out = String::with_capacity(text.len() - (end - start) - trailing_nl);
    out.push_str(&text[..start]);
    out.push_str(&text[end + trailing_nl..]);
    out
}

// ---------- helpers ----------

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let tmp = path.with_extension(format!(
        "{}.tmp",
        path.extension().and_then(|s| s.to_str()).unwrap_or("")
    ));
    std::fs::write(&tmp, bytes).with_context(|| format!("write {}", tmp.display()))?;
    std::fs::rename(&tmp, path)
        .with_context(|| format!("rename {} -> {}", tmp.display(), path.display()))?;
    Ok(())
}

pub fn render_report(r: &InstallReport) -> String {
    let mut out = String::new();
    out.push_str(&format!("dora binary: {}\n", r.binary_path.display()));
    out.push_str(&format!("mcp args:    {}\n\n", r.args.join(" ")));

    let host = |label: &str, (path, action): &(PathBuf, HostAction)| -> String {
        let status = match action {
            HostAction::NotInstalled => format!("not installed ({})", path.display()),
            HostAction::SkippedByFilter => "skipped (--client filter)".to_string(),
            HostAction::AlreadyUpToDate => {
                format!("already up to date ({})", path.display())
            }
            HostAction::Patched => format!("patched ({})", path.display()),
            HostAction::Failed(msg) => format!("FAILED ({}): {msg}", path.display()),
        };
        format!("  {label:<7}  {status}\n")
    };

    out.push_str("MCP hosts:\n");
    out.push_str(&host("Claude", &r.claude));
    out.push_str(&host("Cursor", &r.cursor));
    out.push_str(&host("Codex", &r.codex));

    out.push_str("\nShell wrappers:\n");
    match &r.shell {
        ShellAction::Skipped(msg) => {
            out.push_str(&format!("  skipped: {msg}\n"));
        }
        ShellAction::Done(ops) => {
            if ops.is_empty() {
                out.push_str("  no wrappers requested\n");
            } else {
                let mut any_change = false;
                for (tool, op) in ops {
                    let s = match op {
                        WrapperOp::Added => {
                            any_change = true;
                            format!("  {tool:<6} added to ~/.zshrc")
                        }
                        WrapperOp::Updated => {
                            any_change = true;
                            format!("  {tool:<6} updated in ~/.zshrc")
                        }
                        WrapperOp::Removed => {
                            any_change = true;
                            format!("  {tool:<6} removed from ~/.zshrc")
                        }
                        WrapperOp::AlreadyUpToDate => {
                            format!("  {tool:<6} already up to date")
                        }
                        WrapperOp::UnknownTool => format!(
                            "  {tool:<6} UNKNOWN — supported: {}",
                            SUPPORTED_WRAPS.join(", ")
                        ),
                    };
                    out.push_str(&s);
                    out.push('\n');
                }
                if any_change {
                    out.push_str("  (`source ~/.zshrc` or open a new shell)\n");
                }
            }
        }
    }

    out
}

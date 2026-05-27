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

    // Auto-detect HTTP daemon: if `dora mcp --http --daemon` is currently running, write the
    // HTTP transport shape into each client config instead of the stdio launch command. This
    // lets the persistent server share its loaded models across all MCP clients. If the
    // daemon goes down later, `dora doctor` will surface the mismatch.
    let http_url = detect_http_daemon();
    if let Some(url) = &http_url {
        eprintln!("dora install: detected http daemon at {url} — writing http transport to client configs");
    }

    let claude = if matches!(client, Client::All | Client::Claude) {
        patch_json_host(&claude_path, &binary_path, &args, http_url.as_deref())
    } else {
        HostAction::SkippedByFilter
    };
    let cursor = if matches!(client, Client::All | Client::Cursor) {
        patch_json_host(&cursor_path, &binary_path, &args, http_url.as_deref())
    } else {
        HostAction::SkippedByFilter
    };
    let codex = if matches!(client, Client::All | Client::Codex) {
        patch_toml_host(&codex_path, &binary_path, &args, http_url.as_deref())
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

/// True iff a `dora mcp --http` daemon is currently running and answering /health on the
/// default 127.0.0.1:8181. Returns the URL to wire into client configs.
fn detect_http_daemon() -> Option<String> {
    let pid_path = dirs::home_dir()?
        .join(".config")
        .join("dora")
        .join("mcp-http.pid");
    if !pid_path.exists() {
        return None;
    }
    let pid = std::fs::read_to_string(&pid_path)
        .ok()?
        .trim()
        .parse::<u32>()
        .ok()?;
    let alive = std::process::Command::new("kill")
        .args(["-0", &pid.to_string()])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !alive {
        return None;
    }
    let url = "http://127.0.0.1:8181/mcp";
    // Cheap reachability check — if /health doesn't respond, assume non-default bind and
    // fall back to stdio. The user can rerun `dora install` after `dora mcp stop` if needed.
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_millis(300))
        .build()
        .ok()?;
    client
        .get("http://127.0.0.1:8181/health")
        .send()
        .ok()?
        .error_for_status()
        .ok()?;
    Some(url.to_string())
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

fn patch_json_host(
    path: &Path,
    binary_path: &Path,
    args: &[String],
    http_url: Option<&str>,
) -> HostAction {
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

    let target = if let Some(url) = http_url {
        json!({ "url": url })
    } else {
        json!({
            "command": binary_path.display().to_string(),
            "args": args,
        })
    };

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

fn patch_toml_host(
    path: &Path,
    binary_path: &Path,
    args: &[String],
    http_url: Option<&str>,
) -> HostAction {
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
    if let Some(url) = http_url {
        new_entry.insert("url".to_string(), toml::Value::String(url.to_string()));
    } else {
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
    }
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
# dora {tool} wrapper — routes to dora's semantic search when the invocation is shaped like
# "search this folder for this pattern". Specifically: allow `-r -R -i -n -H` (in any combo
# like `-rin`); take the first non-flag arg as the pattern; treat any subsequent non-flag
# args as paths (default: PWD). If every path is a directory that sits inside a folder with
# `.dora/index.db`, route to `dora "$pattern"` from that source root. Anything else (other
# flags, file paths, paths outside a dora source) falls through to the real {tool}. Disable
# with `dora wrappers off`.
{tool}() {{
    dora wrappers status -q 2>/dev/null || {{ command {tool} "$@"; return; }}
    local _dora_pattern="" _dora_seen=0 _dora_arg _dora_flag _dora_ch
    local -a _dora_paths
    _dora_paths=()
    for _dora_arg in "$@"; do
        case "$_dora_arg" in
            --) command {tool} "$@"; return ;;
            -*)
                _dora_flag="${{_dora_arg#-}}"
                while [ -n "$_dora_flag" ]; do
                    _dora_ch="${{_dora_flag%${{_dora_flag#?}}}}"
                    case "$_dora_ch" in
                        r|R|i|n|H) ;;
                        *) command {tool} "$@"; return ;;
                    esac
                    _dora_flag="${{_dora_flag#?}}"
                done ;;
            *)
                if [ "$_dora_seen" = 0 ]; then
                    _dora_pattern="$_dora_arg"
                    _dora_seen=1
                else
                    _dora_paths+=("$_dora_arg")
                fi ;;
        esac
    done
    [ "$_dora_seen" = 1 ] || {{ command {tool} "$@"; return; }}
    [ ${{#_dora_paths[@]}} -eq 0 ] && _dora_paths=("$PWD")
    local _dora_root="" _dora_p _dora_abs _dora_dir _dora_found
    for _dora_p in "${{_dora_paths[@]}}"; do
        [ -d "$_dora_p" ] || {{ command {tool} "$@"; return; }}
        _dora_abs="$(cd "$_dora_p" 2>/dev/null && pwd)" || {{ command {tool} "$@"; return; }}
        _dora_dir="$_dora_abs"
        _dora_found=""
        while [ "$_dora_dir" != "/" ]; do
            if [ -f "$_dora_dir/.dora/index.db" ]; then
                _dora_found="$_dora_dir"
                break
            fi
            _dora_dir="$(dirname "$_dora_dir")"
        done
        [ -n "$_dora_found" ] || {{ command {tool} "$@"; return; }}
        [ -z "$_dora_root" ] && _dora_root="$_dora_found"
    done
    ( cd "$_dora_root" && dora "$_dora_pattern" )
}}
{end}
"#
    )
}

/// `find` needs a stricter heuristic than grep — its primary use is filesystem traversal
/// with structural predicates (`-name`, `-type`, `-newer`), not content search. Two intercept
/// shapes only, both flagless:
///   1. `find "natural phrase"` — single arg containing whitespace.
///   2. `find <dir> "natural phrase"` — dir + whitespace-containing arg; the path-aware form
///      that mirrors the `grep -r "phrase" <dir>` shape from v0.2.4.
/// Anything else (flags, multiple paths, paths without a quoted phrase) → real find.
fn find_body() -> String {
    let begin = marker_begin("find");
    let end = marker_end("find");
    format!(
        r#"{begin}
# dora find wrapper — intercepts the natural-language shapes
#   find "phrase"                 (PWD must be inside a dora source)
#   find <dir> "phrase"           (dir must resolve into a dora source)
# Both require no flags and the phrase arg to contain whitespace (single-word grep-shaped
# queries belong in `grep`). Anything else → real find. Disable with `dora wrappers off`.
find() {{
    dora wrappers status -q 2>/dev/null || {{ command find "$@"; return; }}
    local _dora_phrase="" _dora_path=""
    if [ "$#" -eq 1 ]; then
        case "$1" in
            -*) command find "$@"; return ;;
            *' '*) _dora_phrase="$1"; _dora_path="$PWD" ;;
            *)    command find "$@"; return ;;
        esac
    elif [ "$#" -eq 2 ]; then
        case "$1" in -*) command find "$@"; return ;; esac
        case "$2" in -*) command find "$@"; return ;; esac
        case "$2" in *' '*) ;; *) command find "$@"; return ;; esac
        [ -d "$1" ] || {{ command find "$@"; return; }}
        _dora_path="$1"
        _dora_phrase="$2"
    else
        command find "$@"; return
    fi
    local _dora_abs _dora_dir _dora_found=""
    _dora_abs="$(cd "$_dora_path" 2>/dev/null && pwd)" || {{ command find "$@"; return; }}
    _dora_dir="$_dora_abs"
    while [ "$_dora_dir" != "/" ]; do
        if [ -f "$_dora_dir/.dora/index.db" ]; then
            _dora_found="$_dora_dir"
            break
        fi
        _dora_dir="$(dirname "$_dora_dir")"
    done
    [ -n "$_dora_found" ] || {{ command find "$@"; return; }}
    ( cd "$_dora_found" && dora "$_dora_phrase" )
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

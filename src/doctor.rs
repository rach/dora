//! `dora doctor` — health report. Reads the registry, each source's DB meta, MCP host configs,
//! and the shell wrapper, and prints a single-screen summary. Exits 1 if anything is `Err`,
//! 0 otherwise (warnings don't fail).

use anyhow::Result;
use std::path::{Path, PathBuf};

use crate::config::{Config, CHUNKER_VERSION, SCHEMA_VERSION};
use crate::registry::Registry;
use crate::store::Store;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckStatus {
    Ok,
    Warn,
    Err,
    Info,
}

#[derive(Debug)]
pub struct Check {
    pub status: CheckStatus,
    pub label: String,
    pub detail: String,
}

#[derive(Debug, Default)]
pub struct DoctorReport {
    pub binary: Vec<Check>,
    pub registry: Vec<Check>,
    pub mcp_hosts: Vec<Check>,
    pub shell: Vec<Check>,
    pub watcher: Vec<Check>,
}

impl DoctorReport {
    pub fn errors(&self) -> usize {
        self.sections()
            .flat_map(|v| v.iter())
            .filter(|c| c.status == CheckStatus::Err)
            .count()
    }
    pub fn warnings(&self) -> usize {
        self.sections()
            .flat_map(|v| v.iter())
            .filter(|c| c.status == CheckStatus::Warn)
            .count()
    }
    fn sections(&self) -> impl Iterator<Item = &Vec<Check>> {
        [
            &self.binary,
            &self.registry,
            &self.mcp_hosts,
            &self.shell,
            &self.watcher,
        ]
        .into_iter()
    }
}

pub fn run() -> Result<DoctorReport> {
    let mut report = DoctorReport::default();
    let home = dirs::home_dir();
    report.binary = check_binary();
    report.registry = check_registry();
    if let Some(ref h) = home {
        report.mcp_hosts = check_mcp_hosts(h);
        report.shell = check_shell_wrapper(h);
    }
    report.watcher = check_watcher();
    Ok(report)
}

// ---------- binary ----------

fn check_binary() -> Vec<Check> {
    let mut out = Vec::new();
    match std::env::current_exe() {
        Ok(p) => {
            out.push(Check {
                status: CheckStatus::Ok,
                label: "binary".into(),
                detail: p.display().to_string(),
            });
        }
        Err(e) => {
            out.push(Check {
                status: CheckStatus::Err,
                label: "binary".into(),
                detail: format!("could not locate current_exe: {e}"),
            });
        }
    }
    out.push(Check {
        status: CheckStatus::Ok,
        label: "version".into(),
        detail: env!("CARGO_PKG_VERSION").to_string(),
    });
    out
}

// ---------- registry ----------

fn check_registry() -> Vec<Check> {
    let mut out = Vec::new();
    let reg = match Registry::load() {
        Ok(r) => r,
        Err(e) => {
            out.push(Check {
                status: CheckStatus::Err,
                label: "registry".into(),
                detail: format!("failed to load: {e}"),
            });
            return out;
        }
    };

    if reg.sources.is_empty() {
        out.push(Check {
            status: CheckStatus::Warn,
            label: "registry".into(),
            detail: "no sources registered — `dora source add <path>`".to_string(),
        });
        return out;
    }

    out.push(Check {
        status: CheckStatus::Info,
        label: "registry".into(),
        detail: format!("{} source(s) registered", reg.sources.len()),
    });

    let now = now_secs();
    for src in &reg.sources {
        let label = format!("  {}", src.name);
        let db = src.path.join(".dora").join("index.db");
        if !db.exists() {
            out.push(Check {
                status: CheckStatus::Err,
                label,
                detail: format!("no .dora/index.db at {} — run `dora index`", src.path.display()),
            });
            continue;
        }

        // Open the DB read-only-ish to check meta. The dim isn't known here; use a placeholder
        // (the CREATE IF NOT EXISTS is harmless on an already-existing schema with any dim).
        // Real check: read meta keys; if present they're authoritative regardless.
        let store = match Store::open(&db, 384) {
            Ok(s) => s,
            Err(e) => {
                out.push(Check {
                    status: CheckStatus::Err,
                    label,
                    detail: format!("can't open db: {e}"),
                });
                continue;
            }
        };

        let meta_schema = store.get_meta("schema_version").ok().flatten();
        let meta_chunker = store.get_meta("chunker_version").ok().flatten();
        let meta_embedder = store.get_meta("embedder_id").ok().flatten();
        let meta_last_walk = store
            .get_meta("last_walk_at")
            .ok()
            .flatten()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);

        // What the current binary + this source's config expect:
        let cfg = Config::load_or_default(&src.path).unwrap_or_default();
        let expected_embedder_prefix = match cfg.embedder.provider.as_str() {
            "fastembed" => format!("fastembed:Xenova/{}", cfg.embedder.model),
            other => format!("{}:{}", other, cfg.embedder.model),
        };

        let mut detail_parts = Vec::new();
        let mut worst = CheckStatus::Ok;

        if meta_schema.as_deref() != Some(SCHEMA_VERSION) {
            detail_parts.push(format!(
                "schema_version mismatch (got {:?}, want {SCHEMA_VERSION})",
                meta_schema
            ));
            worst = CheckStatus::Err;
        }
        if meta_chunker.as_deref() != Some(CHUNKER_VERSION) {
            detail_parts.push(format!(
                "chunker_version mismatch (got {:?}, want {CHUNKER_VERSION})",
                meta_chunker
            ));
            worst = CheckStatus::Err;
        }
        // Embedder canonical id may differ in casing / model_code expansion — we only flag
        // if it's clearly missing or for a different provider, not best-effort.
        match &meta_embedder {
            None => {
                detail_parts.push("embedder_id missing".to_string());
                worst = CheckStatus::Err;
            }
            Some(s) if !s.starts_with(&format!("{}:", cfg.embedder.provider)) => {
                detail_parts.push(format!(
                    "embedder provider mismatch (db has {s:?}, config expects {expected_embedder_prefix:?})"
                ));
                worst = CheckStatus::Err;
            }
            Some(_) => {}
        }

        // Staleness: warn if last walk is older than 7 days.
        let age = now.saturating_sub(meta_last_walk);
        if meta_last_walk == 0 {
            detail_parts.push("never walked".to_string());
            if worst == CheckStatus::Ok {
                worst = CheckStatus::Warn;
            }
        } else if age > 7 * 86_400 {
            let days = age / 86_400;
            detail_parts.push(format!("last walked {days}d ago — `dora index`"));
            if worst == CheckStatus::Ok {
                worst = CheckStatus::Warn;
            }
        }

        // Per-source mode + (for code sources) chunk-kind breakdown. Read directly from
        // the resolved Config and the chunks table — cheap (just COUNT + GROUP BY).
        let mode_str = cfg.source.mode.as_str();
        let kind_breakdown = chunk_kind_breakdown(&store).unwrap_or_default();
        let kind_summary = if kind_breakdown.is_empty() {
            String::new()
        } else {
            // Show non-prose kinds for code sources, else just "prose=N".
            let prose_only = kind_breakdown.iter().all(|(k, _)| k == "prose");
            if prose_only {
                String::new()
            } else {
                let parts: Vec<String> = kind_breakdown
                    .iter()
                    .filter(|(k, _)| k != "prose")
                    .map(|(k, n)| format!("{k}={n}"))
                    .collect();
                format!(", {}", parts.join(" "))
            }
        };
        let link_summary = match store.count_links().ok() {
            Some(n) if n > 0 => format!(", {n} links"),
            _ => String::new(),
        };

        let detail = if detail_parts.is_empty() {
            format!(
                "{}, mode={mode_str}, embedder={}{kind_summary}{link_summary}, walked {}",
                src.path.display(),
                meta_embedder.unwrap_or_default(),
                human_ago(age)
            )
        } else {
            format!(
                "{} (mode={mode_str}) — {}",
                src.path.display(),
                detail_parts.join("; ")
            )
        };

        out.push(Check {
            status: worst,
            label,
            detail,
        });
    }
    out
}

fn chunk_kind_breakdown(store: &Store) -> Result<Vec<(String, i64)>> {
    let mut stmt = store
        .conn()
        .prepare("SELECT kind, COUNT(*) FROM chunks GROUP BY kind ORDER BY COUNT(*) DESC")?;
    let rows = stmt.query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?)))?;
    let mut out = Vec::new();
    for r in rows {
        out.push(r?);
    }
    Ok(out)
}

// ---------- MCP host configs ----------

fn check_mcp_hosts(home: &Path) -> Vec<Check> {
    let mut out = Vec::new();
    out.push(check_json_host(
        "Claude Code",
        &home.join(".claude.json"),
        "mcpServers",
    ));
    out.push(check_json_host(
        "Cursor",
        &home.join(".cursor").join("mcp.json"),
        "mcpServers",
    ));
    out.push(check_toml_host(
        "Codex",
        &home.join(".codex").join("config.toml"),
    ));
    out
}

fn check_json_host(name: &str, path: &Path, key: &str) -> Check {
    if !path.exists() {
        return Check {
            status: CheckStatus::Info,
            label: name.into(),
            detail: format!("{} not found (client not installed?)", path.display()),
        };
    }
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => {
            return Check {
                status: CheckStatus::Err,
                label: name.into(),
                detail: format!("read {}: {e}", path.display()),
            }
        }
    };
    if text.trim().is_empty() {
        return Check {
            status: CheckStatus::Warn,
            label: name.into(),
            detail: format!("{} empty — run `dora install --client {}`", path.display(), name.to_lowercase()),
        };
    }
    let root: serde_json::Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(e) => {
            return Check {
                status: CheckStatus::Err,
                label: name.into(),
                detail: format!("malformed json in {}: {e}", path.display()),
            }
        }
    };
    let has_dora = root
        .get(key)
        .and_then(|v| v.get("dora"))
        .is_some();
    if has_dora {
        Check {
            status: CheckStatus::Ok,
            label: name.into(),
            detail: format!("`dora` registered in {}", path.display()),
        }
    } else {
        Check {
            status: CheckStatus::Warn,
            label: name.into(),
            detail: format!(
                "{} present but no `dora` entry — run `dora install --client {}`",
                path.display(),
                name.to_lowercase().split_whitespace().next().unwrap_or("claude")
            ),
        }
    }
}

fn check_toml_host(name: &str, path: &Path) -> Check {
    if !path.exists() {
        return Check {
            status: CheckStatus::Info,
            label: name.into(),
            detail: format!("{} not found (client not installed?)", path.display()),
        };
    }
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => {
            return Check {
                status: CheckStatus::Err,
                label: name.into(),
                detail: format!("read {}: {e}", path.display()),
            }
        }
    };
    let root: toml::Value = match toml::from_str(&text) {
        Ok(v) => v,
        Err(e) => {
            return Check {
                status: CheckStatus::Err,
                label: name.into(),
                detail: format!("malformed toml in {}: {e}", path.display()),
            }
        }
    };
    let has_dora = root
        .get("mcp_servers")
        .and_then(|v| v.get("dora"))
        .is_some();
    if has_dora {
        Check {
            status: CheckStatus::Ok,
            label: name.into(),
            detail: format!("`dora` registered in {}", path.display()),
        }
    } else {
        Check {
            status: CheckStatus::Warn,
            label: name.into(),
            detail: format!(
                "{} present but no `dora` entry — run `dora install --client codex`",
                path.display()
            ),
        }
    }
}

// ---------- shell ----------

fn check_shell_wrapper(home: &Path) -> Vec<Check> {
    let zshrc = home.join(".zshrc");
    let mut out = Vec::new();
    if !zshrc.exists() {
        out.push(Check {
            status: CheckStatus::Info,
            label: "~/.zshrc".into(),
            detail: "not found — non-zsh shell or never opened a zsh session".to_string(),
        });
        return out;
    }
    let text = std::fs::read_to_string(&zshrc).unwrap_or_default();
    // Scan for any of the supported wrapper marker blocks.
    let installed: Vec<&str> = crate::install::SUPPORTED_WRAPS
        .iter()
        .filter(|tool| text.contains(&format!("# >>> dora {tool} wrapper >>>")))
        .copied()
        .collect();
    if installed.is_empty() {
        out.push(Check {
            status: CheckStatus::Warn,
            label: "~/.zshrc".into(),
            detail: "no dora wrappers installed — run `dora install`".to_string(),
        });
    } else {
        out.push(Check {
            status: CheckStatus::Ok,
            label: "~/.zshrc".into(),
            detail: format!("dora wrappers: {}", installed.join(", ")),
        });
    }
    out
}

// ---------- watcher ----------

fn check_watcher() -> Vec<Check> {
    // Reads ~/.config/dora/watch.pid (written by `dora watch`). Verifies liveness with
    // `kill -0 PID` so a stale PID from a crashed watcher reports correctly. We use a pid
    // file (not pgrep) because pgrep -f always matches the running test harness whose own
    // command line contains the pattern.
    let mut out = Vec::new();
    let pid_path = match crate::watch::pid_file_path() {
        Some(p) => p,
        None => {
            out.push(Check {
                status: CheckStatus::Info,
                label: "watcher".into(),
                detail: "couldn't resolve $HOME/.config/dora/watch.pid".to_string(),
            });
            return out;
        }
    };
    let pid_str = match std::fs::read_to_string(&pid_path) {
        Ok(s) => s.trim().to_string(),
        Err(_) => {
            out.push(Check {
                status: CheckStatus::Info,
                label: "watcher".into(),
                detail: "dora watch not running (optional; queries self-heal otherwise)"
                    .to_string(),
            });
            return out;
        }
    };
    let alive = std::process::Command::new("kill")
        .args(["-0", &pid_str])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if alive {
        out.push(Check {
            status: CheckStatus::Ok,
            label: "watcher".into(),
            detail: format!("dora watch running (pid {pid_str})"),
        });
    } else {
        // Stale PID file — clean it up so we don't keep reporting the same dead pid.
        let _ = std::fs::remove_file(&pid_path);
        out.push(Check {
            status: CheckStatus::Info,
            label: "watcher".into(),
            detail: format!(
                "stale pid file removed (pid {pid_str} not alive); dora watch not running"
            ),
        });
    }
    out
}

// ---------- rendering ----------

pub fn render(r: &DoctorReport) -> String {
    let mut out = String::new();
    out.push_str(&render_section("BINARY", &r.binary));
    out.push_str(&render_section("REGISTRY", &r.registry));
    out.push_str(&render_section("MCP HOSTS", &r.mcp_hosts));
    out.push_str(&render_section("SHELL", &r.shell));
    out.push_str(&render_section("WATCHER", &r.watcher));
    let errs = r.errors();
    let warns = r.warnings();
    out.push_str(&format!(
        "\nResult: {errs} error{}, {warns} warning{}\n",
        plural(errs),
        plural(warns)
    ));
    out
}

fn render_section(title: &str, checks: &[Check]) -> String {
    let mut out = String::new();
    out.push_str(&format!("{title}\n"));
    for c in checks {
        let icon = match c.status {
            CheckStatus::Ok => "✓",
            CheckStatus::Warn => "⚠",
            CheckStatus::Err => "✗",
            CheckStatus::Info => "·",
        };
        out.push_str(&format!("  {icon} {:<22} {}\n", c.label, c.detail));
    }
    out.push('\n');
    out
}

fn plural(n: usize) -> &'static str {
    if n == 1 { "" } else { "s" }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn human_ago(secs: u64) -> String {
    if secs < 60 {
        format!("{secs}s ago")
    } else if secs < 3600 {
        format!("{}m ago", secs / 60)
    } else if secs < 86_400 {
        format!("{}h ago", secs / 3600)
    } else {
        format!("{}d ago", secs / 86_400)
    }
}

/// Suppress unused-warning while keeping this in main.rs's call site clear.
#[allow(dead_code)]
pub fn touch(_: PathBuf) {}

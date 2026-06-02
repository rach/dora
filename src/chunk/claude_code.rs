//! Claude Code session-transcript chunker. Mode = `claude-code`.
//!
//! Claude Code stores each session as `~/.claude/projects/<encoded-cwd>/<uuid>.jsonl` — a
//! stream of typed records (`user`, `assistant`, `system`, `attachment`, plus metadata like
//! `permission-mode`, `ai-title`, etc.). This chunker:
//!   - parses each line as JSON (skipping malformed lines without panicking),
//!   - identifies the project + session metadata from the first record carrying `cwd` /
//!     `gitBranch` / `sessionId`,
//!   - walks records in file order and groups them into per-user-turn chunks: each chunk =
//!     one user prompt + every assistant text/tool block until the next user prompt,
//!   - renders the chunk body as readable prose (`USER:` / `ASSISTANT:` / `[tool: …]` /
//!     `[tool result: …]`),
//!   - skips `thinking` blocks by default (they bloat embeddings without improving recall)
//!     unless `include_thinking = true` in `[claude_code]` config,
//!   - skips `system` / `attachment` records (compaction noise),
//!   - sets `heading_path` to `"<project> · <iso-minute> · <gitBranch>"` so search results
//!     are project-anchored without exposing the ugly encoded folder name.
//!
//! Files whose mtime is too recent (the active session) are filtered out *before* chunking
//! by `run_incremental_index` in main.rs — that's where `settle_seconds` lives. By the time
//! we get here, the file is considered settled.

use serde_json::Value;

use super::{Chunk, ChunkKind, Chunker};
use crate::config::ClaudeCodeConfig;

const TOOL_INPUT_TRUNC: usize = 80;
const TOOL_RESULT_TRUNC: usize = 200;

pub struct ClaudeCodeChunker {
    include_thinking: bool,
}

impl ClaudeCodeChunker {
    pub fn new(cfg: &ClaudeCodeConfig) -> Self {
        Self {
            include_thinking: cfg.include_thinking,
        }
    }
}

impl Chunker for ClaudeCodeChunker {
    fn chunk(&self, text: &str, _rel_path: &str) -> Vec<Chunk> {
        let mut records: Vec<Record> = Vec::new();
        let mut byte_cursor: usize = 0;
        for raw in text.split_inclusive('\n') {
            let line_start = byte_cursor;
            byte_cursor += raw.len();
            let trimmed = raw.trim_end_matches('\n').trim_end_matches('\r');
            if trimmed.is_empty() {
                continue;
            }
            match serde_json::from_str::<Value>(trimmed) {
                Ok(v) => records.push(Record {
                    v,
                    start_byte: line_start,
                    end_byte: byte_cursor,
                }),
                Err(_) => continue,
            }
        }

        // First pass: extract session-wide metadata (project name from `cwd`, branch).
        let mut project_name = String::new();
        let mut git_branch = String::new();
        for r in &records {
            if let Some(cwd) = r.v.get("cwd").and_then(Value::as_str) {
                if project_name.is_empty() {
                    project_name = std::path::Path::new(cwd)
                        .file_name()
                        .map(|s| s.to_string_lossy().to_string())
                        .unwrap_or_else(|| cwd.to_string());
                }
            }
            if let Some(b) = r.v.get("gitBranch").and_then(Value::as_str) {
                if git_branch.is_empty() && !b.is_empty() {
                    git_branch = b.to_string();
                }
            }
            if !project_name.is_empty() && !git_branch.is_empty() {
                break;
            }
        }

        // Second pass: group records into per-user-turn buckets.
        let mut turns: Vec<Turn> = Vec::new();
        let mut current: Option<Turn> = None;
        for r in &records {
            let ty = r.v.get("type").and_then(Value::as_str).unwrap_or("");
            match ty {
                "user" => {
                    if let Some(t) = current.take() {
                        turns.push(t);
                    }
                    let timestamp =
                        r.v.get("timestamp")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string();
                    current = Some(Turn {
                        timestamp,
                        start_byte: r.start_byte,
                        end_byte: r.end_byte,
                        user_text: render_user_message(&r.v),
                        events: Vec::new(),
                    });
                }
                "assistant" => {
                    if let Some(t) = current.as_mut() {
                        t.end_byte = r.end_byte;
                        render_assistant_message(&r.v, self.include_thinking, &mut t.events);
                    }
                    // Assistant before any user message (rare; session metadata only) — drop.
                }
                // Metadata records — not part of the conversation. Skip.
                _ => {}
            }
        }
        if let Some(t) = current.take() {
            turns.push(t);
        }

        // Third pass: render each turn into a Chunk.
        let mut chunks = Vec::with_capacity(turns.len());
        for (idx, turn) in turns.into_iter().enumerate() {
            if turn.user_text.is_empty() && turn.events.is_empty() {
                continue;
            }
            let mut body = String::new();
            if !turn.user_text.is_empty() {
                body.push_str("USER: ");
                body.push_str(&turn.user_text);
                body.push('\n');
            }
            if !turn.events.is_empty() {
                body.push('\n');
                body.push_str("ASSISTANT:\n");
                for ev in &turn.events {
                    body.push_str(ev);
                    body.push('\n');
                }
            }

            let heading_path = build_heading(&project_name, &turn.timestamp, &git_branch);
            chunks.push(Chunk {
                idx,
                heading_path,
                content: body,
                start_byte: turn.start_byte,
                end_byte: turn.end_byte,
                kind: ChunkKind::Prose,
                symbol: None,
                parent_chunk_idx: None,
            });
        }
        chunks
    }
}

// ---------- internal types ----------

struct Record {
    v: Value,
    start_byte: usize,
    end_byte: usize,
}

struct Turn {
    timestamp: String,
    start_byte: usize,
    end_byte: usize,
    user_text: String,
    events: Vec<String>,
}

// ---------- render helpers ----------

/// Extract the user-prompt text from a `user` record. The `message.content` field is either
/// a string (normal prompt) or a list of blocks (tool result echo, attachments). We flatten
/// the list form to `[tool result: …]` so it still contributes signal without dragging in
/// raw JSON.
fn render_user_message(v: &Value) -> String {
    let Some(msg) = v.get("message") else {
        return String::new();
    };
    let Some(content) = msg.get("content") else {
        return String::new();
    };
    match content {
        Value::String(s) => s.trim().to_string(),
        Value::Array(blocks) => {
            let mut out = Vec::new();
            for b in blocks {
                let bt = b.get("type").and_then(Value::as_str).unwrap_or("");
                match bt {
                    "text" => {
                        if let Some(t) = b.get("text").and_then(Value::as_str) {
                            out.push(t.trim().to_string());
                        }
                    }
                    "tool_result" => {
                        let result = b
                            .get("content")
                            .map(stringify_tool_content)
                            .unwrap_or_default();
                        out.push(format!(
                            "[tool result: {}]",
                            truncate(&result, TOOL_RESULT_TRUNC)
                        ));
                    }
                    _ => {}
                }
            }
            out.join("\n")
        }
        _ => String::new(),
    }
}

/// Render an `assistant` record into one or more `[tool: …]` / `[tool result: …]` lines and
/// text-block paragraphs. Appends each into the caller's events buffer.
fn render_assistant_message(v: &Value, include_thinking: bool, out: &mut Vec<String>) {
    let Some(msg) = v.get("message") else {
        return;
    };
    let Some(content) = msg.get("content") else {
        return;
    };
    let Value::Array(blocks) = content else {
        return;
    };
    for b in blocks {
        let bt = b.get("type").and_then(Value::as_str).unwrap_or("");
        match bt {
            "text" => {
                if let Some(t) = b.get("text").and_then(Value::as_str) {
                    let trimmed = t.trim();
                    if !trimmed.is_empty() {
                        out.push(trimmed.to_string());
                    }
                }
            }
            "thinking" if include_thinking => {
                if let Some(t) = b.get("thinking").and_then(Value::as_str) {
                    let trimmed = t.trim();
                    if !trimmed.is_empty() {
                        out.push(format!("[thinking] {trimmed}"));
                    }
                }
            }
            "tool_use" => {
                let name = b.get("name").and_then(Value::as_str).unwrap_or("?");
                let input_summary = b.get("input").map(summarize_tool_input).unwrap_or_default();
                if input_summary.is_empty() {
                    out.push(format!("[tool: {name}]"));
                } else {
                    out.push(format!(
                        "[tool: {name} {}]",
                        truncate(&input_summary, TOOL_INPUT_TRUNC)
                    ));
                }
            }
            // Some sessions embed `tool_result` blocks directly on the assistant message
            // (less common; usually they're on the next `user` record).
            "tool_result" => {
                let result = b
                    .get("content")
                    .map(stringify_tool_content)
                    .unwrap_or_default();
                out.push(format!(
                    "[tool result: {}]",
                    truncate(&result, TOOL_RESULT_TRUNC)
                ));
            }
            _ => {}
        }
    }
}

/// Squash a tool-input JSON value into a short readable line. Prefers the most informative
/// scalar field (file_path, command, query, pattern) over generic JSON.
fn summarize_tool_input(input: &Value) -> String {
    if let Some(obj) = input.as_object() {
        for key in ["file_path", "command", "query", "pattern", "path", "url"] {
            if let Some(s) = obj.get(key).and_then(Value::as_str) {
                if !s.is_empty() {
                    return s.to_string();
                }
            }
        }
    }
    // Fallback: stringify and let truncate cap it.
    input.to_string()
}

/// `tool_result.content` is sometimes a string, sometimes an array of `{type:"text", text:"…"}`
/// blocks. Flatten to a single string.
fn stringify_tool_content(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(parts) => {
            let mut out = String::new();
            for p in parts {
                if let Some(t) = p.get("text").and_then(Value::as_str) {
                    if !out.is_empty() {
                        out.push(' ');
                    }
                    out.push_str(t);
                }
            }
            out
        }
        _ => v.to_string(),
    }
}

fn truncate(s: &str, max: usize) -> String {
    let collapsed: String = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() <= max {
        return collapsed;
    }
    let mut out: String = collapsed.chars().take(max).collect();
    out.push('…');
    out
}

fn build_heading(project: &str, timestamp: &str, branch: &str) -> String {
    let mut parts: Vec<String> = Vec::new();
    if !project.is_empty() {
        parts.push(project.to_string());
    }
    // Truncate ISO timestamp like "2026-05-24T18:50:16.091Z" to "2026-05-24 18:50".
    let short_ts = timestamp
        .split_once('T')
        .map(|(date, rest)| {
            let hm = rest.split(':').take(2).collect::<Vec<_>>().join(":");
            format!("{date} {hm}")
        })
        .unwrap_or_else(|| timestamp.to_string());
    if !short_ts.is_empty() {
        parts.push(short_ts);
    }
    if !branch.is_empty() {
        parts.push(format!("branch:{branch}"));
    }
    parts.join(" · ")
}

// ---------- tests ----------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chunker(thinking: bool) -> ClaudeCodeChunker {
        ClaudeCodeChunker::new(&ClaudeCodeConfig {
            include_thinking: thinking,
            settle_seconds: 60,
        })
    }

    #[test]
    fn three_turns_yields_three_chunks() {
        let jsonl = concat!(
            r#"{"type":"user","timestamp":"2026-05-25T10:00:00Z","cwd":"/Users/me/Dev/myproj","gitBranch":"main","message":{"role":"user","content":"first prompt"}}"#,
            "\n",
            r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"first reply"}]}}"#,
            "\n",
            r#"{"type":"user","timestamp":"2026-05-25T10:01:00Z","cwd":"/Users/me/Dev/myproj","message":{"role":"user","content":"second prompt"}}"#,
            "\n",
            r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"second reply"},{"type":"tool_use","name":"Read","input":{"file_path":"/Users/me/foo.rs"}}]}}"#,
            "\n",
            r#"{"type":"user","timestamp":"2026-05-25T10:02:00Z","cwd":"/Users/me/Dev/myproj","message":{"role":"user","content":"third prompt"}}"#,
            "\n",
            r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"third reply"}]}}"#,
            "\n",
        );
        let chunker = make_chunker(false);
        let chunks = chunker.chunk(jsonl, "myproj/session.jsonl");
        assert_eq!(chunks.len(), 3);
        assert!(chunks[0].content.contains("USER: first prompt"));
        assert!(chunks[0].content.contains("first reply"));
        assert!(chunks[1].content.contains("USER: second prompt"));
        assert!(chunks[1].content.contains("[tool: Read /Users/me/foo.rs]"));
        assert!(chunks[2].content.contains("third reply"));
        // heading_path carries project + timestamp
        for c in &chunks {
            assert!(
                c.heading_path.starts_with("myproj"),
                "got: {}",
                c.heading_path
            );
            assert!(c.heading_path.contains("2026-05-25"));
            assert!(c.heading_path.contains("branch:main"));
        }
    }

    #[test]
    fn thinking_blocks_skipped_by_default() {
        let jsonl = concat!(
            r#"{"type":"user","timestamp":"2026-05-25T10:00:00Z","cwd":"/x","message":{"role":"user","content":"hi"}}"#,
            "\n",
            r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"thinking","thinking":"INTERNAL_SECRET_PLAN"},{"type":"text","text":"visible reply"}]}}"#,
            "\n",
        );
        let chunks = make_chunker(false).chunk(jsonl, "x.jsonl");
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].content.contains("visible reply"));
        assert!(!chunks[0].content.contains("INTERNAL_SECRET_PLAN"));
    }

    #[test]
    fn thinking_blocks_included_when_opted_in() {
        let jsonl = r#"{"type":"user","timestamp":"2026-05-25T10:00:00Z","cwd":"/x","message":{"role":"user","content":"hi"}}
{"type":"assistant","message":{"role":"assistant","content":[{"type":"thinking","thinking":"REASONING"},{"type":"text","text":"reply"}]}}
"#;
        let chunks = make_chunker(true).chunk(jsonl, "x.jsonl");
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].content.contains("REASONING"));
    }

    #[test]
    fn malformed_lines_are_skipped() {
        let jsonl = concat!(
            "not json\n",
            r#"{"type":"user","timestamp":"2026-05-25T10:00:00Z","cwd":"/x","message":{"role":"user","content":"hi"}}"#,
            "\n",
            "{also not json\n",
            r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"yo"}]}}"#,
            "\n",
        );
        let chunks = make_chunker(false).chunk(jsonl, "x.jsonl");
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].content.contains("yo"));
    }

    #[test]
    fn system_and_attachment_records_skipped() {
        let jsonl = concat!(
            r#"{"type":"attachment","attachment":{"foo":"bar"}}"#,
            "\n",
            r#"{"type":"permission-mode","permissionMode":"auto"}"#,
            "\n",
            r#"{"type":"user","timestamp":"2026-05-25T10:00:00Z","cwd":"/x","message":{"role":"user","content":"hello"}}"#,
            "\n",
            r#"{"type":"system","content":"a system reminder"}"#,
            "\n",
            r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"hi back"}]}}"#,
            "\n",
        );
        let chunks = make_chunker(false).chunk(jsonl, "x.jsonl");
        assert_eq!(chunks.len(), 1);
        assert!(!chunks[0].content.contains("system reminder"));
        assert!(chunks[0].content.contains("USER: hello"));
        assert!(chunks[0].content.contains("hi back"));
    }
}

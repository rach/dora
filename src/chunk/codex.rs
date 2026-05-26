//! OpenAI Codex CLI session-transcript chunker. Mode = `codex`.
//!
//! Codex stores each session as `~/.codex/sessions/YYYY/MM/DD/rollout-<iso>-<uuid>.jsonl`.
//! Every line is an envelope `{timestamp, type, payload}`:
//!   - `session_meta`              — first line, payload carries `cwd`, `cli_version`, `timestamp`
//!   - `response_item` + `message` — user/assistant prompts; payload.content is a list of
//!     `input_text` (user) / `output_text` (assistant) blocks
//!   - `response_item` + `reasoning` — Codex's analog of Claude's `thinking`
//!   - `response_item` + `function_call` — tool use; arguments is a JSON-encoded STRING
//!   - `response_item` + `function_call_output` — tool result, linked by call_id
//!   - `event_msg` / `turn_context` — status records, skipped
//!
//! Mirrors the per-user-turn shape of `claude_code.rs`: each chunk = one user prompt + every
//! assistant/tool record until the next user prompt. heading_path = `<project> · <iso-min> · codex`.
//! The trailing `· codex` distinguishes Codex hits from Claude Code hits in cross-source
//! search results.

use serde_json::Value;
use std::collections::HashMap;

use super::{Chunk, ChunkKind, Chunker};
use crate::config::CodexConfig;

const TOOL_INPUT_TRUNC: usize = 80;
const TOOL_RESULT_TRUNC: usize = 200;

pub struct CodexChunker {
    include_reasoning: bool,
}

impl CodexChunker {
    pub fn new(cfg: &CodexConfig) -> Self {
        Self {
            include_reasoning: cfg.include_reasoning,
        }
    }
}

impl Chunker for CodexChunker {
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

        // First pass: session metadata from session_meta line.
        let mut project_name = String::new();
        for r in &records {
            if r.v.get("type").and_then(Value::as_str) == Some("session_meta") {
                if let Some(cwd) = r
                    .v
                    .get("payload")
                    .and_then(|p| p.get("cwd"))
                    .and_then(Value::as_str)
                {
                    project_name = std::path::Path::new(cwd)
                        .file_name()
                        .map(|s| s.to_string_lossy().to_string())
                        .unwrap_or_else(|| cwd.to_string());
                }
                break;
            }
        }
        // Tool-name lookup for function_call_output (which only carries call_id).
        let mut tool_names: HashMap<String, String> = HashMap::new();
        for r in &records {
            let Some(p) = r.v.get("payload") else { continue };
            if p.get("type").and_then(Value::as_str) == Some("function_call") {
                if let (Some(call_id), Some(name)) = (
                    p.get("call_id").and_then(Value::as_str),
                    p.get("name").and_then(Value::as_str),
                ) {
                    tool_names.insert(call_id.to_string(), name.to_string());
                }
            }
        }

        // Second pass: walk records, group by user-turn.
        let mut turns: Vec<Turn> = Vec::new();
        let mut current: Option<Turn> = None;
        for r in &records {
            let ty = r.v.get("type").and_then(Value::as_str).unwrap_or("");
            if ty != "response_item" {
                continue; // skip session_meta, event_msg, turn_context
            }
            let Some(p) = r.v.get("payload") else { continue };
            let p_type = p.get("type").and_then(Value::as_str).unwrap_or("");

            // User message → open a new turn.
            if p_type == "message" && p.get("role").and_then(Value::as_str) == Some("user") {
                if let Some(t) = current.take() {
                    turns.push(t);
                }
                let timestamp = r
                    .v
                    .get("timestamp")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let user_text = render_message_content(p);
                current = Some(Turn {
                    timestamp,
                    start_byte: r.start_byte,
                    end_byte: r.end_byte,
                    user_text,
                    events: Vec::new(),
                });
                continue;
            }

            // Everything else belongs to the current turn.
            let Some(turn) = current.as_mut() else {
                continue;
            };
            turn.end_byte = r.end_byte;

            match p_type {
                "message" => {
                    // assistant text
                    let txt = render_message_content(p);
                    if !txt.is_empty() {
                        turn.events.push(txt);
                    }
                }
                "reasoning" => {
                    if self.include_reasoning {
                        let summary = render_reasoning(p);
                        if !summary.is_empty() {
                            turn.events
                                .push(format!("[reasoning] {}", truncate(&summary, TOOL_RESULT_TRUNC)));
                        }
                    }
                }
                "function_call" => {
                    let name = p.get("name").and_then(Value::as_str).unwrap_or("?");
                    let summary = summarize_function_args(p);
                    if summary.is_empty() {
                        turn.events.push(format!("[tool: {name}]"));
                    } else {
                        turn.events.push(format!(
                            "[tool: {name} {}]",
                            truncate(&summary, TOOL_INPUT_TRUNC)
                        ));
                    }
                }
                "function_call_output" => {
                    let call_id = p.get("call_id").and_then(Value::as_str).unwrap_or("");
                    let name = tool_names
                        .get(call_id)
                        .cloned()
                        .unwrap_or_else(|| "?".to_string());
                    let output = p
                        .get("output")
                        .and_then(Value::as_str)
                        .map(strip_output_metadata)
                        .unwrap_or_default();
                    turn.events.push(format!(
                        "[tool result {name}: {}]",
                        truncate(&output, TOOL_RESULT_TRUNC)
                    ));
                }
                _ => {} // unknown payload type
            }
        }
        if let Some(t) = current.take() {
            turns.push(t);
        }

        // Third pass: render to Chunks.
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
            chunks.push(Chunk {
                idx,
                heading_path: build_heading(&project_name, &turn.timestamp),
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

// ---------- internal ----------

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

/// Concatenate the text inside a `message` payload's content array (`input_text` for user,
/// `output_text` for assistant).
fn render_message_content(payload: &Value) -> String {
    let Some(blocks) = payload.get("content").and_then(Value::as_array) else {
        return String::new();
    };
    let mut parts: Vec<String> = Vec::new();
    for b in blocks {
        let bt = b.get("type").and_then(Value::as_str).unwrap_or("");
        match bt {
            "input_text" | "output_text" | "text" => {
                if let Some(t) = b.get("text").and_then(Value::as_str) {
                    let trimmed = t.trim();
                    if !trimmed.is_empty() {
                        parts.push(trimmed.to_string());
                    }
                }
            }
            _ => {}
        }
    }
    parts.join("\n")
}

/// Extract a readable summary from a `reasoning` payload. The block shape varies; common
/// shapes: `{summary: [{type:"summary_text", text:"…"}]}` or `{content: [{type:"reasoning_text", text:"…"}]}`.
fn render_reasoning(payload: &Value) -> String {
    let candidates = ["summary", "content"];
    for key in candidates {
        if let Some(blocks) = payload.get(key).and_then(Value::as_array) {
            let mut parts: Vec<String> = Vec::new();
            for b in blocks {
                if let Some(t) = b.get("text").and_then(Value::as_str) {
                    let trimmed = t.trim();
                    if !trimmed.is_empty() {
                        parts.push(trimmed.to_string());
                    }
                }
            }
            if !parts.is_empty() {
                return parts.join(" ");
            }
        }
    }
    String::new()
}

/// `function_call.arguments` is a JSON-encoded STRING (not parsed). Parse it, then pick the
/// most informative scalar field. Fall back to the raw arguments string.
fn summarize_function_args(payload: &Value) -> String {
    let Some(args_str) = payload.get("arguments").and_then(Value::as_str) else {
        return String::new();
    };
    if let Ok(parsed) = serde_json::from_str::<Value>(args_str) {
        if let Some(obj) = parsed.as_object() {
            for key in [
                "cmd",
                "command",
                "file_path",
                "path",
                "query",
                "pattern",
                "url",
            ] {
                if let Some(s) = obj.get(key).and_then(Value::as_str) {
                    if !s.is_empty() {
                        return s.to_string();
                    }
                }
            }
        }
    }
    args_str.to_string()
}

/// Codex prepends `function_call_output` payloads with metadata lines like:
///   `Chunk ID: …\nWall time: 0.0521 seconds\nProcess exited with code 0\nOriginal token count: 428\nOutput:\n<real output>`
/// Strip everything up to and including the first `"Output:\n"` marker. If absent, return as-is.
fn strip_output_metadata(s: &str) -> String {
    const MARKER: &str = "Output:\n";
    if let Some(pos) = s.find(MARKER) {
        s[pos + MARKER.len()..].to_string()
    } else {
        s.to_string()
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

fn build_heading(project: &str, timestamp: &str) -> String {
    let mut parts: Vec<String> = Vec::new();
    if !project.is_empty() {
        parts.push(project.to_string());
    }
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
    parts.push("codex".to_string());
    parts.join(" · ")
}

// ---------- tests ----------

#[cfg(test)]
mod tests {
    use super::*;

    fn make(include_reasoning: bool) -> CodexChunker {
        CodexChunker::new(&CodexConfig {
            include_reasoning,
            settle_seconds: 60,
        })
    }

    #[test]
    fn three_turns_yields_three_chunks() {
        let jsonl = concat!(
            r#"{"timestamp":"2026-03-07T20:21:20.607Z","type":"session_meta","payload":{"id":"abc","timestamp":"2026-03-07T20:21:20.607Z","cwd":"/Users/me/Dev/myproj","cli_version":"0.110.0"}}"#, "\n",
            r#"{"timestamp":"2026-03-07T20:21:22.425Z","type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"first prompt"}]}}"#, "\n",
            r#"{"timestamp":"2026-03-07T20:21:23.000Z","type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"first reply"}]}}"#, "\n",
            r#"{"timestamp":"2026-03-07T20:22:00.000Z","type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"second prompt"}]}}"#, "\n",
            r#"{"timestamp":"2026-03-07T20:22:01.000Z","type":"response_item","payload":{"type":"function_call","name":"exec_command","arguments":"{\"cmd\":\"ls -la\"}","call_id":"call_1"}}"#, "\n",
            r#"{"timestamp":"2026-03-07T20:22:02.000Z","type":"response_item","payload":{"type":"function_call_output","call_id":"call_1","output":"Chunk ID: x\nWall time: 0.01 seconds\nProcess exited with code 0\nOutput:\ntotal 808\ndrwxr-xr-x"}}"#, "\n",
            r#"{"timestamp":"2026-03-07T20:23:00.000Z","type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"third prompt"}]}}"#, "\n",
            r#"{"timestamp":"2026-03-07T20:23:01.000Z","type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"third reply"}]}}"#, "\n",
        );
        let chunks = make(false).chunk(jsonl, "myproj/rollout-...jsonl");
        assert_eq!(chunks.len(), 3);
        assert!(chunks[0].content.contains("USER: first prompt"));
        assert!(chunks[0].content.contains("first reply"));
        assert!(chunks[1].content.contains("USER: second prompt"));
        assert!(chunks[1].content.contains("[tool: exec_command ls -la]"));
        assert!(
            chunks[1].content.contains("[tool result exec_command:"),
            "got: {}",
            chunks[1].content
        );
        assert!(
            chunks[1].content.contains("total 808"),
            "metadata header should be stripped, got: {}",
            chunks[1].content
        );
        assert!(
            !chunks[1].content.contains("Wall time"),
            "metadata header should be stripped, got: {}",
            chunks[1].content
        );
        assert!(chunks[2].content.contains("third reply"));
        for c in &chunks {
            assert!(c.heading_path.starts_with("myproj"), "got: {}", c.heading_path);
            assert!(c.heading_path.contains("2026-03-07"));
            assert!(c.heading_path.ends_with("· codex"));
        }
    }

    #[test]
    fn reasoning_blocks_skipped_by_default() {
        let jsonl = concat!(
            r#"{"timestamp":"2026-03-07T20:21:20Z","type":"session_meta","payload":{"cwd":"/x"}}"#, "\n",
            r#"{"timestamp":"2026-03-07T20:21:22Z","type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}}"#, "\n",
            r#"{"timestamp":"2026-03-07T20:21:23Z","type":"response_item","payload":{"type":"reasoning","summary":[{"type":"summary_text","text":"INTERNAL_SECRET_PLAN"}]}}"#, "\n",
            r#"{"timestamp":"2026-03-07T20:21:24Z","type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"visible reply"}]}}"#, "\n",
        );
        let chunks = make(false).chunk(jsonl, "x.jsonl");
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].content.contains("visible reply"));
        assert!(!chunks[0].content.contains("INTERNAL_SECRET_PLAN"));
    }

    #[test]
    fn reasoning_blocks_included_when_opted_in() {
        let jsonl = concat!(
            r#"{"timestamp":"2026-03-07T20:21:20Z","type":"session_meta","payload":{"cwd":"/x"}}"#, "\n",
            r#"{"timestamp":"2026-03-07T20:21:22Z","type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}}"#, "\n",
            r#"{"timestamp":"2026-03-07T20:21:23Z","type":"response_item","payload":{"type":"reasoning","summary":[{"type":"summary_text","text":"REASONING_DETAIL"}]}}"#, "\n",
            r#"{"timestamp":"2026-03-07T20:21:24Z","type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"reply"}]}}"#, "\n",
        );
        let chunks = make(true).chunk(jsonl, "x.jsonl");
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].content.contains("REASONING_DETAIL"));
    }

    #[test]
    fn malformed_lines_are_skipped() {
        let jsonl = concat!(
            "not json\n",
            r#"{"timestamp":"2026-03-07T20:21:20Z","type":"session_meta","payload":{"cwd":"/x"}}"#, "\n",
            "{also broken\n",
            r#"{"timestamp":"2026-03-07T20:21:22Z","type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"yo"}]}}"#, "\n",
            r#"{"timestamp":"2026-03-07T20:21:23Z","type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hi"}]}}"#, "\n",
        );
        let chunks = make(false).chunk(jsonl, "x.jsonl");
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].content.contains("USER: yo"));
        assert!(chunks[0].content.contains("hi"));
    }

    #[test]
    fn event_msg_and_turn_context_skipped() {
        let jsonl = concat!(
            r#"{"timestamp":"2026-03-07T20:21:20Z","type":"session_meta","payload":{"cwd":"/x"}}"#, "\n",
            r#"{"timestamp":"2026-03-07T20:21:21Z","type":"event_msg","payload":{"type":"task_started"}}"#, "\n",
            r#"{"timestamp":"2026-03-07T20:21:21Z","type":"turn_context","payload":{}}"#, "\n",
            r#"{"timestamp":"2026-03-07T20:21:22Z","type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}}"#, "\n",
            r#"{"timestamp":"2026-03-07T20:21:23Z","type":"event_msg","payload":{"type":"token_count","total":1234}}"#, "\n",
            r#"{"timestamp":"2026-03-07T20:21:24Z","type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hello"}]}}"#, "\n",
        );
        let chunks = make(false).chunk(jsonl, "x.jsonl");
        assert_eq!(chunks.len(), 1);
        assert!(!chunks[0].content.contains("task_started"));
        assert!(!chunks[0].content.contains("token_count"));
        assert!(chunks[0].content.contains("USER: hi"));
        assert!(chunks[0].content.contains("hello"));
    }

    #[test]
    fn output_metadata_passthrough_when_no_marker() {
        // When the Output: marker is absent, the raw output text passes through unchanged.
        assert_eq!(strip_output_metadata("raw output here"), "raw output here");
        assert_eq!(
            strip_output_metadata("Chunk ID: x\nOutput:\nactual"),
            "actual"
        );
    }
}

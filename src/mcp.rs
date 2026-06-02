//! Stdio MCP server exposing two tools: `search` and `list_sources`.
//!
//! Multi-source by construction. A registry of N sources becomes a HashMap<name, SourceState>.
//! The single-source case (`dora mcp --source <path>`) is just a registry-of-one — same code path.
//!
//! Embedders are cached by `(provider, model, dimensions)` during boot: sources that share
//! the same model share a single Arc'd Embedder instance, saving ~80 MB per duplicate.

use anyhow::{Context, Result};
use rmcp::{
    model::{
        CallToolRequestParams, CallToolResult, Content, Implementation, ListToolsResult,
        PaginatedRequestParams, ServerCapabilities, ServerInfo, Tool, ToolAnnotations,
    },
    service::{MaybeSendFuture, RequestContext},
    transport::stdio,
    ErrorData, RoleServer, ServerHandler, ServiceExt,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::chunk::{self, Chunker};
use crate::config::Config;
use crate::embed::{self, DynEmbedder};
use crate::registry::{Registry, Source};
use crate::store::Store;
use crate::{check_meta, db_path, models_dir};

#[derive(Debug, Deserialize, JsonSchema)]
struct SearchArgs {
    /// Free-text query. Hybrid FTS5 + vector ANN search merged via Reciprocal Rank Fusion.
    query: String,
    /// Source name to scope the search to. If omitted, dora searches every registered source
    /// and merges the results by score.
    #[serde(default)]
    source: Option<String>,
    /// Maximum number of hits to return. Default 10, capped at 50. Ignored when `all=true`.
    #[serde(default)]
    top_k: Option<u32>,
    /// Path-prefix filter applied within each searched source (e.g. "Daily/").
    #[serde(default)]
    path_prefix: Option<String>,
    /// Drop hits whose RRF score is below this threshold. Combine with `all` for
    /// "every relevant document above this confidence" flows.
    #[serde(default)]
    min_score: Option<f64>,
    /// Disable the top_k cap and return every hit that passed `min_score` (if set, else
    /// every hit). Useful with `output: "files"` to enumerate every matching file.
    #[serde(default)]
    all: Option<bool>,
    /// Output mode. "chunks" (default) returns one hit per chunk with snippet + line.
    /// "files" dedupes by path and returns one hit per file (line=0, no snippet).
    #[serde(default)]
    output: Option<String>,
    /// Boolean intersection terms. Each must also score for a chunk to remain a result;
    /// combined score is the harmonic mean of normalized per-query scores. Equivalent to
    /// `dora "query" --and "Y" --and "Z"` on the CLI. Use to narrow a too-broad result.
    #[serde(default)]
    and: Option<Vec<String>>,
    /// Boolean exclusion terms. Chunks scoring highly on any are dropped; weaker matches
    /// get a soft demote. Equivalent to `dora "query" --not "Z"`. Use to filter out a
    /// known distractor topic from a search.
    #[serde(default)]
    not: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ListSourcesArgs {}

#[derive(Debug, Deserialize, JsonSchema)]
struct FindDefinitionArgs {
    /// Symbol name (function/struct/class/trait/etc.) to find the definition of.
    symbol: String,
    /// Source to scope the search to. Omit to search every code source.
    #[serde(default)]
    source: Option<String>,
    /// Cap on returned definitions. Default 10, max 50.
    #[serde(default)]
    limit: Option<u32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct FindCallersArgs {
    /// Symbol whose callers you want. Returns chunks that call this function/method.
    symbol: String,
    #[serde(default)]
    source: Option<String>,
    /// Transitive depth. 1 = direct callers only (default); >1 walks the call graph.
    #[serde(default)]
    depth: Option<u32>,
    #[serde(default)]
    limit: Option<u32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct FindImplementationsArgs {
    /// Trait / interface name to find implementations of.
    symbol: String,
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    limit: Option<u32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct RepoMapArgs {
    /// Source name. Required — repo_map is per-source (PageRank scores aren't comparable
    /// across separately-computed graphs).
    source: String,
    /// File path prefixes the agent is currently focused on. PageRank biases ranking toward
    /// files matching these prefixes (50× weight). Pass the files you're editing.
    #[serde(default)]
    focus_paths: Vec<String>,
    /// Approximate token budget for the rendered outline. Default 2000.
    #[serde(default)]
    token_budget: Option<u32>,
}

#[derive(Debug, Serialize)]
struct DefinitionResult {
    chunk_id: i64,
    path: String,
    heading_path: String,
    symbol: String,
    kind: String,
    content: String,
    start_byte: usize,
}

#[derive(Debug, Serialize)]
struct CallerResult {
    chunk_id: i64,
    path: String,
    heading_path: String,
    symbol: String,
    kind: String,
    content: String,
    start_byte: usize,
    distance: usize,
    confidence: String,
}

#[derive(Debug, Serialize)]
struct RepoMapResult {
    source: String,
    focus_paths: Vec<String>,
    outline: String,
    file_count: usize,
    chunk_count: usize,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct BacklinksArgs {
    /// Source name (from `list_sources`).
    source: String,
    /// Note path relative to the source root, e.g. `Projects/dora.md`.
    path: String,
}

#[derive(Debug, Serialize)]
struct BacklinksResult {
    source: String,
    path: String,
    /// Notes that link TO this note (inbound / backlinks).
    inbound: Vec<String>,
    /// Notes this note links to (outbound).
    outbound: Vec<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct MultiGetArgs {
    /// Source name (from `list_sources`) to scope the glob against.
    source: String,
    /// Glob pattern relative to the source root, e.g. `src/**/*.rs`, `notes/2026-*.md`,
    /// `docs/README.md`. Matches `files.path` exactly via the `globset` crate's semantics.
    pattern: String,
    /// Per-file byte cap. Files larger than this are truncated and flagged. Default 102400
    /// (~25k tokens per file at 4 chars/token).
    #[serde(default)]
    max_bytes: Option<u64>,
}

#[derive(Debug, Serialize)]
struct MultiGetEntry {
    path: String,
    content: String,
    byte_count: usize,
    truncated: bool,
}

#[derive(Debug, Serialize)]
struct MultiGetResult {
    source: String,
    pattern: String,
    entries: Vec<MultiGetEntry>,
}

#[derive(Debug, Serialize)]
struct SourceInfo {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    path: String,
    embedder_id: String,
    file_count: i64,
    chunk_count: i64,
    last_indexed_at: Option<i64>,
}

struct SourceState {
    name: String,
    description: Option<String>,
    path: PathBuf,
    cfg: Config,
    embedder: DynEmbedder,
    chunker: Box<dyn Chunker>,
    store: Store,
}

/// Maximum gap between a `search` and a follow-up `multi_get` for the latter to count as
/// "this is the search result the user/agent acted on." Matches the PRD's 60s window.
const USE_ATTRIBUTION_WINDOW_SECS: i64 = 60;
/// Cap on the recent-search ring buffer. Above this, oldest entries get dropped. 64 is
/// enough headroom for a small handful of concurrent clients with bursty traffic.
const RECENT_SEARCHES_CAP: usize = 64;

/// In-memory ring buffer of recent searches across all sources/clients. Each entry records
/// the source, query string, and the set of file paths returned. A subsequent `multi_get`
/// that reads any of those paths attributes the read back to the most recent matching
/// search via `Store::mark_used_by_query`. Best-effort — no correctness criticality.
struct RecentSearch {
    source: String,
    query: String,
    /// Paths in this source returned by the search. Keep as Vec — search results are small.
    paths: Vec<String>,
    ts: i64,
}

#[derive(Default)]
struct MultiSourceState {
    sources: HashMap<String, SourceState>,
    /// Stable ordered list of source names — matches insertion order for deterministic
    /// `list_sources` output and predictable cross-source iteration.
    order: Vec<String>,
    /// Recent searches indexed for use-attribution. Push on every `search` call, scan on
    /// every `multi_get` call. Bounded by `RECENT_SEARCHES_CAP`.
    recent_searches: std::collections::VecDeque<RecentSearch>,
}

#[derive(Clone)]
struct DoraServer {
    state: Arc<Mutex<MultiSourceState>>,
}

impl ServerHandler for DoraServer {
    fn get_info(&self) -> ServerInfo {
        let instructions = self.state.lock().ok().map(|s| build_instructions(&s));

        let mut impl_info = Implementation::new("dora", env!("CARGO_PKG_VERSION"));
        impl_info.title = Some("dora — local semantic memory".to_string());
        impl_info.description = Some(
            "Local-first semantic search across registered markdown sources (notes, code, \
             transcripts). Indexes incrementally; results are always fresh."
                .to_string(),
        );

        let mut info = ServerInfo::new(ServerCapabilities::builder().enable_tools().build());
        info.server_info = impl_info;
        info.instructions = instructions;
        info
    }

    fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<ListToolsResult, ErrorData>> + MaybeSendFuture + '_
    {
        let state = self.state.clone();
        async move {
            // Build the source-listing snippet for the `search` tool's `source` parameter
            // description so agents see registered sources (+ their descriptions) inline,
            // no separate list_sources round-trip required.
            let source_summary = {
                let guard = state.lock().map_err(|e| {
                    ErrorData::internal_error(format!("state lock poisoned: {e}"), None)
                })?;
                source_summary_text(&guard)
            };

            let search_schema = build_search_schema(&source_summary);
            let search = Tool::new(
                "search",
                "Semantic search across one or all registered sources. Hybrid FTS5 + vector \
                 hits ranked via Reciprocal Rank Fusion. Each hit carries the source it came \
                 from. Omit `source` to merge results across every registered source.",
                std::sync::Arc::new(search_schema),
            )
            .annotate(ToolAnnotations::new().read_only(true));

            let list = Tool::new(
                "list_sources",
                "List every source dora is currently serving — name, optional description, \
                 path, embedder id, and file/chunk counts. Useful when the `search` tool's \
                 inline source list is truncated or you want exact counts.",
                std::sync::Arc::new(serde_json::Map::new()),
            )
            .with_input_schema::<ListSourcesArgs>()
            .annotate(ToolAnnotations::new().read_only(true));

            let find_def = Tool::new(
                "find_definition",
                "Locate the definition(s) of a symbol (function/struct/class/trait/interface). \
                 Works in code sources (Rust, Python, TS/JS, Go, Java). Returns the chunk \
                 containing the definition. Prefer this over `search` when you know the exact \
                 name.",
                std::sync::Arc::new(serde_json::Map::new()),
            )
            .with_input_schema::<FindDefinitionArgs>()
            .annotate(ToolAnnotations::new().read_only(true));

            let find_callers = Tool::new(
                "find_callers",
                "Find chunks that call a given function/method. `depth` walks the call graph \
                 transitively (default 1 = direct callers only, max 5). Each result carries a \
                 `confidence`: 'exact' (within-file or unique cross-file name match) or \
                 'name_match' (ambiguous — multiple definitions of the same name).",
                std::sync::Arc::new(serde_json::Map::new()),
            )
            .with_input_schema::<FindCallersArgs>()
            .annotate(ToolAnnotations::new().read_only(true));

            let find_impls = Tool::new(
                "find_implementations",
                "Find implementations of a trait/interface (Rust `impl Trait for ...`, Java/TS \
                 `implements`). Returns the chunks that contain the implementing methods.",
                std::sync::Arc::new(serde_json::Map::new()),
            )
            .with_input_schema::<FindImplementationsArgs>()
            .annotate(ToolAnnotations::new().read_only(true));

            let repo_map = Tool::new(
                "repo_map",
                "Ranked outline of a code source via PageRank over the symbol graph. Pass \
                 `focus_paths` (file path prefixes you're currently editing) to bias the \
                 ranking toward neighbors of those files. Renders a flat outline (path:line: \
                 signature) up to `token_budget`. Use this to give an agent an at-a-glance \
                 picture of which code matters for the current task.",
                std::sync::Arc::new(serde_json::Map::new()),
            )
            .with_input_schema::<RepoMapArgs>()
            .annotate(ToolAnnotations::new().read_only(true));

            let multi_get = Tool::new(
                "multi_get",
                "Batch-retrieve documents by glob pattern relative to a registered source's \
                 root (e.g. `src/**/*.rs`, `notes/2026-*.md`). Returns body text per match, \
                 truncated at `max_bytes` (default 102400). Use this instead of N×Read when \
                 you already know which files you want — saves round-trips for the agent.",
                std::sync::Arc::new(serde_json::Map::new()),
            )
            .with_input_schema::<MultiGetArgs>()
            .annotate(ToolAnnotations::new().read_only(true));

            let backlinks = Tool::new(
                "backlinks",
                "Show the wikilink graph for a note: which notes link TO it (inbound / \
                 backlinks) and which it links to (outbound). Built from `[[wikilinks]]` and \
                 `[text](note.md)` links at index time. Use this to discover related notes and \
                 navigate a vault structurally, the way Obsidian's backlinks pane does.",
                std::sync::Arc::new(serde_json::Map::new()),
            )
            .with_input_schema::<BacklinksArgs>()
            .annotate(ToolAnnotations::new().read_only(true));

            Ok(ListToolsResult {
                tools: vec![
                    search,
                    list,
                    find_def,
                    find_callers,
                    find_impls,
                    repo_map,
                    multi_get,
                    backlinks,
                ],
                ..Default::default()
            })
        }
    }

    fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<CallToolResult, ErrorData>> + MaybeSendFuture + '_
    {
        let state = self.state.clone();
        async move {
            match request.name.as_ref() {
                "search" => handle_search(&state, request).await,
                "list_sources" => handle_list_sources(&state).await,
                "find_definition" => handle_find_definition(&state, request).await,
                "find_callers" => handle_find_callers(&state, request).await,
                "find_implementations" => handle_find_implementations(&state, request).await,
                "repo_map" => handle_repo_map(&state, request).await,
                "multi_get" => handle_multi_get(&state, request).await,
                "backlinks" => handle_backlinks(&state, request).await,
                other => Err(ErrorData::invalid_params(
                    format!("unknown tool: {other}"),
                    None,
                )),
            }
        }
    }
}

async fn handle_search(
    state: &Arc<Mutex<MultiSourceState>>,
    request: CallToolRequestParams,
) -> Result<CallToolResult, ErrorData> {
    let args_value = request
        .arguments
        .map(serde_json::Value::Object)
        .unwrap_or(serde_json::Value::Null);
    let args: SearchArgs = serde_json::from_value(args_value)
        .map_err(|e| ErrorData::invalid_params(format!("bad arguments: {e}"), None))?;

    if args.query.trim().is_empty() {
        return Err(ErrorData::invalid_params(
            "query must not be empty".to_string(),
            None,
        ));
    }
    let top_k = args.top_k.unwrap_or(10).clamp(1, 50) as usize;
    let output = match args.output.as_deref() {
        Some("files") => crate::search::OutputMode::Files,
        Some("chunks") | None => crate::search::OutputMode::Chunks,
        Some(other) => {
            return Err(ErrorData::invalid_params(
                format!("invalid output mode {other:?} (expected 'chunks' or 'files')"),
                None,
            ));
        }
    };
    let opts = crate::search::SearchOptions {
        top_k,
        min_score: args.min_score,
        all: args.all.unwrap_or(false),
        path_prefix: args.path_prefix.as_deref(),
        output,
        and_queries: args.and.unwrap_or_default(),
        not_queries: args.not.unwrap_or_default(),
        diagnostics: false,
    };

    let mut guard = state
        .lock()
        .map_err(|e| ErrorData::internal_error(format!("state lock poisoned: {e}"), None))?;
    let multi = &mut *guard;

    let hits = match args.source {
        Some(name) => {
            if !multi.sources.contains_key(&name) {
                let available = multi.order.join(", ");
                return Err(ErrorData::invalid_params(
                    format!("unknown source '{name}'. registered: {available}"),
                    None,
                ));
            }
            let s = multi.sources.get_mut(&name).expect("checked above");
            search_one(s, &args.query, opts.clone())
                .map_err(|e| ErrorData::internal_error(format!("search failed: {e}"), None))?
        }
        None => search_cross(multi, &args.query, opts.clone()).map_err(|e| {
            ErrorData::internal_error(format!("cross-source search failed: {e}"), None)
        })?,
    };

    // Stash this search in the ring buffer so a subsequent `multi_get` can attribute reads
    // back to it. Each `Hit` already knows its source name (set by `search::search`); we
    // group by that so cross-source searches still attribute correctly.
    let mut by_source: HashMap<String, Vec<String>> = HashMap::new();
    for h in &hits {
        by_source
            .entry(h.source.clone())
            .or_default()
            .push(h.path.clone());
    }
    for (src, paths) in by_source {
        multi.record_search(&src, &args.query, paths);
    }

    let json = serde_json::to_string(&hits)
        .map_err(|e| ErrorData::internal_error(format!("serialize hits: {e}"), None))?;
    Ok(CallToolResult::success(vec![Content::text(json)]))
}

async fn handle_list_sources(
    state: &Arc<Mutex<MultiSourceState>>,
) -> Result<CallToolResult, ErrorData> {
    let mut guard = state
        .lock()
        .map_err(|e| ErrorData::internal_error(format!("state lock poisoned: {e}"), None))?;
    let multi = &mut *guard;

    let mut out: Vec<SourceInfo> = Vec::with_capacity(multi.order.len());
    for name in multi.order.clone() {
        let s = multi.sources.get(&name).expect("name in order map exists");
        let file_count = s
            .store
            .count_files()
            .map_err(|e| ErrorData::internal_error(format!("count_files({name}): {e}"), None))?;
        let chunk_count = s
            .store
            .count_chunks()
            .map_err(|e| ErrorData::internal_error(format!("count_chunks({name}): {e}"), None))?;
        let last_indexed_at = s
            .store
            .get_meta("last_walk_at")
            .ok()
            .flatten()
            .and_then(|v| v.parse::<i64>().ok());
        out.push(SourceInfo {
            name: s.name.clone(),
            description: s.description.clone(),
            path: s.path.display().to_string(),
            embedder_id: s.embedder.id().to_string(),
            file_count,
            chunk_count,
            last_indexed_at,
        });
    }

    let json = serde_json::to_string(&out)
        .map_err(|e| ErrorData::internal_error(format!("serialize: {e}"), None))?;
    Ok(CallToolResult::success(vec![Content::text(json)]))
}

// ---------------- code-aware handlers ----------------

fn parse_args<T: for<'de> serde::Deserialize<'de>>(
    request: CallToolRequestParams,
) -> Result<T, ErrorData> {
    let v = request
        .arguments
        .map(serde_json::Value::Object)
        .unwrap_or(serde_json::Value::Null);
    serde_json::from_value(v)
        .map_err(|e| ErrorData::invalid_params(format!("bad arguments: {e}"), None))
}

async fn handle_find_definition(
    state: &Arc<Mutex<MultiSourceState>>,
    request: CallToolRequestParams,
) -> Result<CallToolResult, ErrorData> {
    let args: FindDefinitionArgs = parse_args(request)?;
    let limit = args.limit.unwrap_or(10).clamp(1, 50) as usize;
    let symbol = args.symbol.trim().to_string();
    if symbol.is_empty() {
        return Err(ErrorData::invalid_params(
            "symbol must not be empty".to_string(),
            None,
        ));
    }

    let mut guard = state
        .lock()
        .map_err(|e| ErrorData::internal_error(format!("state lock poisoned: {e}"), None))?;

    let names = match &args.source {
        Some(n) => {
            if !guard.sources.contains_key(n) {
                return Err(ErrorData::invalid_params(
                    format!(
                        "unknown source '{n}'. registered: {}",
                        guard.order.join(", ")
                    ),
                    None,
                ));
            }
            vec![n.clone()]
        }
        None => guard.order.clone(),
    };

    let mut results: Vec<DefinitionResult> = Vec::new();
    for name in names {
        let s = guard.sources.get_mut(&name).expect("name in order");
        match s.store.find_definitions(&symbol, limit) {
            Ok(hits) => {
                for h in hits {
                    results.push(DefinitionResult {
                        chunk_id: h.chunk_id,
                        path: format!("{}/{}", name, h.path),
                        heading_path: h.heading_path,
                        symbol: h.symbol,
                        kind: h.kind,
                        content: h.content,
                        start_byte: h.start_byte,
                    });
                }
            }
            Err(e) => eprintln!("dora mcp: find_definition({name}, {symbol}): {e}"),
        }
        if results.len() >= limit {
            results.truncate(limit);
            break;
        }
    }

    let json = serde_json::to_string(&results)
        .map_err(|e| ErrorData::internal_error(format!("serialize: {e}"), None))?;
    Ok(CallToolResult::success(vec![Content::text(json)]))
}

async fn handle_find_callers(
    state: &Arc<Mutex<MultiSourceState>>,
    request: CallToolRequestParams,
) -> Result<CallToolResult, ErrorData> {
    let args: FindCallersArgs = parse_args(request)?;
    let limit = args.limit.unwrap_or(20).clamp(1, 100) as usize;
    let depth = args.depth.unwrap_or(1).clamp(1, 5) as usize;
    let symbol = args.symbol.trim().to_string();
    if symbol.is_empty() {
        return Err(ErrorData::invalid_params(
            "symbol must not be empty".to_string(),
            None,
        ));
    }

    let mut guard = state
        .lock()
        .map_err(|e| ErrorData::internal_error(format!("state lock poisoned: {e}"), None))?;

    let names = match &args.source {
        Some(n) => {
            if !guard.sources.contains_key(n) {
                return Err(ErrorData::invalid_params(
                    format!(
                        "unknown source '{n}'. registered: {}",
                        guard.order.join(", ")
                    ),
                    None,
                ));
            }
            vec![n.clone()]
        }
        None => guard.order.clone(),
    };

    let mut results: Vec<CallerResult> = Vec::new();
    for name in names {
        let s = guard.sources.get_mut(&name).expect("name in order");
        match s.store.find_callers(&symbol, depth, limit) {
            Ok(hits) => {
                for h in hits {
                    results.push(CallerResult {
                        chunk_id: h.chunk_id,
                        path: format!("{}/{}", name, h.path),
                        heading_path: h.heading_path,
                        symbol: h.symbol,
                        kind: h.kind,
                        content: h.content,
                        start_byte: h.start_byte,
                        distance: h.distance,
                        confidence: h.confidence,
                    });
                }
            }
            Err(e) => eprintln!("dora mcp: find_callers({name}, {symbol}): {e}"),
        }
    }
    results.truncate(limit);

    let json = serde_json::to_string(&results)
        .map_err(|e| ErrorData::internal_error(format!("serialize: {e}"), None))?;
    Ok(CallToolResult::success(vec![Content::text(json)]))
}

async fn handle_find_implementations(
    state: &Arc<Mutex<MultiSourceState>>,
    request: CallToolRequestParams,
) -> Result<CallToolResult, ErrorData> {
    let args: FindImplementationsArgs = parse_args(request)?;
    let limit = args.limit.unwrap_or(20).clamp(1, 100) as usize;
    let symbol = args.symbol.trim().to_string();
    if symbol.is_empty() {
        return Err(ErrorData::invalid_params(
            "symbol must not be empty".to_string(),
            None,
        ));
    }

    let mut guard = state
        .lock()
        .map_err(|e| ErrorData::internal_error(format!("state lock poisoned: {e}"), None))?;

    let names = match &args.source {
        Some(n) => {
            if !guard.sources.contains_key(n) {
                return Err(ErrorData::invalid_params(
                    format!(
                        "unknown source '{n}'. registered: {}",
                        guard.order.join(", ")
                    ),
                    None,
                ));
            }
            vec![n.clone()]
        }
        None => guard.order.clone(),
    };

    let mut results: Vec<DefinitionResult> = Vec::new();
    for name in names {
        let s = guard.sources.get_mut(&name).expect("name in order");
        match s.store.find_implementations(&symbol, limit) {
            Ok(hits) => {
                for h in hits {
                    results.push(DefinitionResult {
                        chunk_id: h.chunk_id,
                        path: format!("{}/{}", name, h.path),
                        heading_path: h.heading_path,
                        symbol: h.symbol,
                        kind: h.kind,
                        content: h.content,
                        start_byte: h.start_byte,
                    });
                }
            }
            Err(e) => eprintln!("dora mcp: find_implementations({name}, {symbol}): {e}"),
        }
    }
    results.truncate(limit);

    let json = serde_json::to_string(&results)
        .map_err(|e| ErrorData::internal_error(format!("serialize: {e}"), None))?;
    Ok(CallToolResult::success(vec![Content::text(json)]))
}

async fn handle_repo_map(
    state: &Arc<Mutex<MultiSourceState>>,
    request: CallToolRequestParams,
) -> Result<CallToolResult, ErrorData> {
    let args: RepoMapArgs = parse_args(request)?;
    let token_budget = args.token_budget.unwrap_or(2000).clamp(200, 10_000) as usize;

    let mut guard = state
        .lock()
        .map_err(|e| ErrorData::internal_error(format!("state lock poisoned: {e}"), None))?;
    if !guard.sources.contains_key(&args.source) {
        return Err(ErrorData::invalid_params(
            format!(
                "unknown source '{}'. registered: {}",
                args.source,
                guard.order.join(", ")
            ),
            None,
        ));
    }
    let s = guard.sources.get_mut(&args.source).expect("checked above");

    let ranks = crate::pagerank::compute(s.store.conn(), &args.focus_paths).map_err(|e| {
        ErrorData::internal_error(format!("pagerank failed for {}: {e}", args.source), None)
    })?;

    // Sort files by rank descending, pull definitions for the top files until we exhaust
    // the token budget. ~4 chars per token is the conventional rough estimate.
    let mut ranked: Vec<(i64, f64)> = ranks.into_iter().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let file_ids: Vec<i64> = ranked.iter().map(|(id, _)| *id).collect();
    let mut outline = String::new();
    let mut chunks_emitted = 0usize;
    let mut files_emitted: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();
    let char_budget = token_budget * 4;

    let entries = s
        .store
        .definitions_in_files(&file_ids)
        .map_err(|e| ErrorData::internal_error(format!("definitions_in_files: {e}"), None))?;
    let by_file: std::collections::HashMap<i64, Vec<&crate::store::OutlineEntry>> = entries
        .iter()
        .fold(std::collections::HashMap::new(), |mut acc, e| {
            acc.entry(e.file_id).or_default().push(e);
            acc
        });

    for (file_id, score) in &ranked {
        let Some(file_defs) = by_file.get(file_id) else {
            continue;
        };
        let header = format!("\n{} (score {:.4})\n", file_defs[0].path, score);
        if outline.len() + header.len() > char_budget {
            break;
        }
        outline.push_str(&header);
        files_emitted.insert(*file_id);
        for def in file_defs {
            let kind_short = match def.kind.as_str() {
                "function" => "fn",
                "method" => "fn",
                "class" => "class",
                "struct" => "struct",
                "trait" => "trait",
                "interface" => "iface",
                "module" => "mod",
                "const" => "const",
                "enum" => "enum",
                "macro" => "macro!",
                _ => &def.kind,
            };
            let qualifier = if def.heading_path.is_empty() {
                String::new()
            } else {
                format!("{}::", def.heading_path)
            };
            let line = format!("  {kind_short} {qualifier}{}\n", def.symbol);
            if outline.len() + line.len() > char_budget {
                break;
            }
            outline.push_str(&line);
            chunks_emitted += 1;
        }
        if outline.len() >= char_budget {
            break;
        }
    }

    let result = RepoMapResult {
        source: args.source.clone(),
        focus_paths: args.focus_paths.clone(),
        outline,
        file_count: files_emitted.len(),
        chunk_count: chunks_emitted,
    };
    let json = serde_json::to_string(&result)
        .map_err(|e| ErrorData::internal_error(format!("serialize: {e}"), None))?;
    Ok(CallToolResult::success(vec![Content::text(json)]))
}

async fn handle_multi_get(
    state: &Arc<Mutex<MultiSourceState>>,
    request: CallToolRequestParams,
) -> Result<CallToolResult, ErrorData> {
    let args: MultiGetArgs = parse_args(request)?;
    let max_bytes = args.max_bytes.unwrap_or(102_400) as usize;

    let mut guard = state
        .lock()
        .map_err(|e| ErrorData::internal_error(format!("state lock poisoned: {e}"), None))?;
    let multi = &mut *guard;
    let s = multi.sources.get(&args.source).ok_or_else(|| {
        ErrorData::invalid_params(
            format!(
                "unknown source '{}'. registered: {}",
                args.source,
                multi.order.join(", ")
            ),
            None,
        )
    })?;
    let paths = s
        .store
        .list_paths_matching(&args.pattern)
        .map_err(|e| ErrorData::internal_error(format!("glob lookup: {e}"), None))?;
    let mut entries = Vec::with_capacity(paths.len());
    for rel in paths {
        let full = s.path.join(&rel);
        let Ok(body) = std::fs::read_to_string(&full) else {
            // Silently skip files that disappeared / aren't UTF-8 — partial result is more
            // useful to the agent than failing the whole call.
            continue;
        };
        let byte_count = body.len();
        let (content, truncated) = if byte_count > max_bytes {
            // Find a char boundary at or just below max_bytes so we don't slice a codepoint.
            let mut cut = max_bytes;
            while cut > 0 && !body.is_char_boundary(cut) {
                cut -= 1;
            }
            (body[..cut].to_string(), true)
        } else {
            (body, false)
        };
        entries.push(MultiGetEntry {
            path: rel,
            content,
            byte_count,
            truncated,
        });
    }

    // Attribute these reads back to any recent in-window search that returned them. For
    // each read path, find the most recent matching `RecentSearch` and patch the usage row.
    attribute_reads(multi, &args.source, &entries);

    let result = MultiGetResult {
        source: args.source,
        pattern: args.pattern,
        entries,
    };
    let json = serde_json::to_string(&result)
        .map_err(|e| ErrorData::internal_error(format!("serialize: {e}"), None))?;
    Ok(CallToolResult::success(vec![Content::text(json)]))
}

/// For each entry read by a `multi_get` call, scan the ring buffer for the most recent
/// in-window search (same source) whose results included that path. If found, look up the
/// chunk_id of the path in the source's index and patch the matching `usage` row's
/// `used_chunk_id`. Best-effort: any failure here just means the read goes unattributed.
fn attribute_reads(multi: &mut MultiSourceState, source: &str, entries: &[MultiGetEntry]) {
    let cutoff = now_secs() - USE_ATTRIBUTION_WINDOW_SECS;
    // Drop expired entries first so the scan is bounded.
    while let Some(front) = multi.recent_searches.front() {
        if front.ts < cutoff {
            multi.recent_searches.pop_front();
        } else {
            break;
        }
    }
    let Some(s) = multi.sources.get(source) else {
        return;
    };
    for entry in entries {
        // Walk newest-first so we always attribute to the most recent matching search.
        let matching = multi
            .recent_searches
            .iter()
            .rev()
            .find(|rs| rs.source == source && rs.paths.iter().any(|p| p == &entry.path));
        let Some(rs) = matching else { continue };
        // Resolve chunk_id for this path. multi_get returns whole files; the search returned
        // chunks. Pick the first chunk of the file as the canonical "this is what was read"
        // attribution — good enough for v0.7's signal collection (we're learning at the
        // file level anyway, since the user/agent always reads whole files via multi_get).
        let chunk_id = match chunk_id_for_path(&s.store, &entry.path) {
            Ok(Some(id)) => id,
            _ => continue,
        };
        if let Err(err) =
            s.store
                .mark_used_by_query(&rs.query, chunk_id, USE_ATTRIBUTION_WINDOW_SECS)
        {
            eprintln!("dora mcp: mark_used_by_query failed: {err}");
        }
    }
}

fn chunk_id_for_path(store: &Store, path: &str) -> Result<Option<i64>> {
    let mut stmt = store.conn().prepare(
        "SELECT c.id FROM chunks c JOIN files f ON f.id = c.file_id \
         WHERE f.path = ? ORDER BY c.chunk_idx LIMIT 1",
    )?;
    let mut rows = stmt.query(rusqlite::params![path])?;
    if let Some(row) = rows.next()? {
        Ok(Some(row.get::<_, i64>(0)?))
    } else {
        Ok(None)
    }
}

async fn handle_backlinks(
    state: &Arc<Mutex<MultiSourceState>>,
    request: CallToolRequestParams,
) -> Result<CallToolResult, ErrorData> {
    let args: BacklinksArgs = parse_args(request)?;
    let guard = state
        .lock()
        .map_err(|e| ErrorData::internal_error(format!("state lock poisoned: {e}"), None))?;
    let s = guard.sources.get(&args.source).ok_or_else(|| {
        ErrorData::invalid_params(
            format!(
                "unknown source '{}'. registered: {}",
                args.source,
                guard.order.join(", ")
            ),
            None,
        )
    })?;
    let inbound = s
        .store
        .backlinks(&args.path)
        .map_err(|e| ErrorData::internal_error(format!("backlinks: {e}"), None))?;
    let outbound = s
        .store
        .forward_links(&args.path)
        .map_err(|e| ErrorData::internal_error(format!("forward_links: {e}"), None))?;
    let result = BacklinksResult {
        source: args.source,
        path: args.path,
        inbound,
        outbound,
    };
    let json = serde_json::to_string(&result)
        .map_err(|e| ErrorData::internal_error(format!("serialize: {e}"), None))?;
    Ok(CallToolResult::success(vec![Content::text(json)]))
}

fn search_one(
    s: &mut SourceState,
    query: &str,
    opts: crate::search::SearchOptions<'_>,
) -> Result<Vec<crate::search::Hit>> {
    crate::search_with_self_heal(
        crate::SearchRuntime {
            source_root: &s.path,
            source_name: &s.name,
            cfg: &s.cfg,
            store: &mut s.store,
            chunker: &s.chunker,
            embedder: s.embedder.as_ref(),
        },
        query,
        opts,
    )
}

/// Cross-source: over-fetch per source (2× top_k when capped), then re-sort + apply the
/// caller's `top_k` / `all` / `min_score` on the merged list. RRF scores are on the same
/// scale (1/(60+rank)) so direct comparison is defensible.
fn search_cross(
    multi: &mut MultiSourceState,
    query: &str,
    opts: crate::search::SearchOptions<'_>,
) -> Result<Vec<crate::search::Hit>> {
    // Per-source over-fetch only when we'd otherwise truncate. With `all`, fetch every hit.
    let per_source_opts = if opts.all {
        opts.clone()
    } else {
        crate::search::SearchOptions {
            top_k: (opts.top_k * 2).max(opts.top_k),
            ..opts.clone()
        }
    };
    let mut all: Vec<crate::search::Hit> = Vec::new();
    for name in multi.order.clone() {
        let s = multi.sources.get_mut(&name).expect("name in order");
        match search_one(s, query, per_source_opts.clone()) {
            Ok(hits) => all.extend(hits),
            Err(e) => {
                // Don't fail the whole call because one source errored — log + continue.
                eprintln!("dora mcp: source '{name}' search failed: {e}");
            }
        }
    }
    all.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if !opts.all {
        all.truncate(opts.top_k);
    }
    Ok(all)
}

// ---------------- entrypoints ----------------

/// Pick the wire format the MCP server speaks on. stdio is the default (one subprocess per
/// client, baked into the v0 install path). HTTP is the v0.5 addition for the
/// many-clients-one-process deployment.
#[derive(Debug, Clone)]
pub enum Transport {
    Stdio,
    Http { bind: std::net::SocketAddr },
}

/// Multi-source MCP server. Loads every source in the registry, sharing embedder instances
/// across sources that pick the same `(provider, model, dimensions)`.
pub async fn run_multi(registry: Registry, transport: Transport) -> Result<()> {
    let mut state = build_state(&registry.sources)?;
    if state.sources.is_empty() {
        anyhow::bail!("no sources could be loaded — check logs above for per-source errors");
    }
    log_ready(&state);
    let shared = Arc::new(Mutex::new(std::mem::take(&mut state)));

    match transport {
        Transport::Stdio => {
            let server = DoraServer { state: shared };
            server
                .serve(stdio())
                .await
                .context("serve stdio MCP transport")?
                .waiting()
                .await
                .context("waiting on MCP service")?;
            Ok(())
        }
        Transport::Http { bind } => run_http(shared, bind).await,
    }
}

/// HTTP transport. Mounts rmcp's `StreamableHttpService` at `/mcp` plus a tiny `/health`
/// JSON endpoint that `dora doctor` + the daemon-detection helpers in `install.rs` use.
/// State (the `Arc<Mutex<MultiSourceState>>`) is captured by the service factory closure,
/// so all sessions share one MultiSourceState — embedders + Stores stay resident.
async fn run_http(state: Arc<Mutex<MultiSourceState>>, bind: std::net::SocketAddr) -> Result<()> {
    use axum::routing::get;
    use rmcp::transport::streamable_http_server::{
        session::local::LocalSessionManager, tower::StreamableHttpServerConfig,
        tower::StreamableHttpService,
    };

    let started_at = std::time::Instant::now();
    let session_manager = Arc::new(LocalSessionManager::default());
    let mut config = StreamableHttpServerConfig::default();
    config.stateful_mode = false;

    let factory_state = state.clone();
    let svc = StreamableHttpService::new(
        move || {
            Ok(DoraServer {
                state: factory_state.clone(),
            })
        },
        session_manager,
        config,
    );

    let health_state = state.clone();
    let app = axum::Router::new()
        .route(
            "/health",
            get(move || {
                let state = health_state.clone();
                async move { axum::Json(health_payload(&state, started_at)) }
            }),
        )
        .nest_service("/mcp", svc);

    let listener = tokio::net::TcpListener::bind(bind)
        .await
        .with_context(|| format!("bind {}", bind))?;
    eprintln!("dora mcp: http listening on http://{}", bind);
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            eprintln!("dora mcp: SIGINT received, draining...");
        })
        .await
        .context("axum http serve")?;
    Ok(())
}

#[derive(Serialize)]
struct HealthPayload {
    status: &'static str,
    version: &'static str,
    uptime_secs: u64,
    sources: Vec<String>,
}

fn health_payload(
    state: &Arc<Mutex<MultiSourceState>>,
    started_at: std::time::Instant,
) -> HealthPayload {
    let sources = state
        .lock()
        .ok()
        .map(|g| g.order.clone())
        .unwrap_or_default();
    HealthPayload {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
        uptime_secs: started_at.elapsed().as_secs(),
        sources,
    }
}

/// Single-source MCP server (used by `dora mcp --source <path>`). Builds a synthetic
/// one-entry registry and reuses `run_multi`'s implementation. Always stdio — single-source
/// is a one-off interactive path; HTTP daemon is meant for the multi-source registry.
pub async fn run(source_root: &Path) -> Result<()> {
    let source_root = source_root
        .canonicalize()
        .context("canonicalize source path")?;
    let name = source_root
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "source".to_string());
    let registry = Registry {
        sources: vec![Source {
            name,
            path: source_root,
            description: None,
        }],
    };
    run_multi(registry, Transport::Stdio).await
}

fn build_state(sources: &[Source]) -> Result<MultiSourceState> {
    let mut cache: HashMap<String, DynEmbedder> = HashMap::new();
    let mut map: HashMap<String, SourceState> = HashMap::new();
    let mut order: Vec<String> = Vec::with_capacity(sources.len());

    for src in sources {
        match try_load_source(src, &mut cache) {
            Ok(state) => {
                order.push(src.name.clone());
                map.insert(src.name.clone(), state);
            }
            Err(e) => {
                eprintln!(
                    "dora mcp: skipping source '{}' ({}): {e}",
                    src.name,
                    src.path.display()
                );
            }
        }
    }

    Ok(MultiSourceState {
        sources: map,
        order,
        recent_searches: std::collections::VecDeque::new(),
    })
}

fn try_load_source(src: &Source, cache: &mut HashMap<String, DynEmbedder>) -> Result<SourceState> {
    let path = src
        .path
        .canonicalize()
        .with_context(|| format!("canonicalize {}", src.path.display()))?;
    let cfg = Config::load_or_default(&path).context("load config")?;
    let key = embed::cache_key(&cfg.embedder);
    let embedder = match cache.get(&key) {
        Some(e) => e.clone(),
        None => {
            let new = embed::from_config(&cfg.embedder, &models_dir(&path))?;
            cache.insert(key, new.clone());
            new
        }
    };
    let chunker = chunk::from_config(&cfg, &path);

    let db = db_path(&path);
    if !db.exists() {
        anyhow::bail!(
            ".dora/index.db not found — run `dora index {}` first",
            path.display()
        );
    }
    let store = Store::open(&db, embedder.dims())?;
    check_meta(&store, embedder.as_ref())?;

    Ok(SourceState {
        name: src.name.clone(),
        description: src.description.clone(),
        path,
        cfg,
        embedder,
        chunker,
        store,
    })
}

impl MultiSourceState {
    /// Push a fresh search into the ring buffer, dropping the oldest entry once we exceed
    /// the cap. Called from `handle_search` after a successful search.
    fn record_search(&mut self, source: &str, query: &str, paths: Vec<String>) {
        if paths.is_empty() {
            return;
        }
        self.recent_searches.push_back(RecentSearch {
            source: source.to_string(),
            query: query.to_string(),
            paths,
            ts: now_secs(),
        });
        while self.recent_searches.len() > RECENT_SEARCHES_CAP {
            self.recent_searches.pop_front();
        }
    }
}

fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Server-level `instructions` shown to MCP clients (Claude Code/Cursor/Codex) on initialize.
/// Distinct from per-tool descriptions: this is the agent's first read of what this *server*
/// is for — useful when multiple `dora-*` MCP servers are registered with different `--include`
/// scopes.
fn build_instructions(state: &MultiSourceState) -> String {
    let n = state.order.len();
    let names = state.order.join(", ");
    let plural = if n == 1 { "source" } else { "sources" };
    format!(
        "dora is serving {n} {plural}: {names}.\n\n\
         Per-source descriptions are inlined in the `search` tool's `source` parameter \
         description. Call `list_sources` for exact file/chunk counts.\n\n\
         The index is incremental and self-healing: every `search` call diffs the underlying \
         vault if file mtimes changed since the last walk, so results always reflect on-disk \
         content. No need to ask the user to re-index after edits."
    )
}

fn source_summary_text(state: &MultiSourceState) -> String {
    if state.order.is_empty() {
        return "(no sources loaded)".to_string();
    }
    let mut lines = Vec::with_capacity(state.order.len());
    for name in &state.order {
        let s = state.sources.get(name).expect("name in order map exists");
        match &s.description {
            Some(d) => lines.push(format!("  - {}: {}", s.name, d)),
            None => lines.push(format!("  - {}", s.name)),
        }
    }
    lines.join("\n")
}

fn build_search_schema(source_summary: &str) -> serde_json::Map<String, serde_json::Value> {
    let source_desc = format!(
        "Which source to scope the search to. Omit to search across every registered source \
         and merge results by score. Registered sources:\n{source_summary}"
    );
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Free-text query. Hybrid FTS5 + vector ANN search merged via Reciprocal Rank Fusion."
            },
            "source": {
                "type": "string",
                "description": source_desc,
            },
            "top_k": {
                "type": "integer",
                "minimum": 1,
                "maximum": 50,
                "default": 10,
                "description": "Maximum number of hits to return. Defaults to 10, capped at 50."
            },
            "path_prefix": {
                "type": "string",
                "description": "Path-prefix filter applied within each searched source (e.g. \"Daily/\")."
            },
            "min_score": {
                "type": "number",
                "description": "Drop hits whose merged RRF score is below this threshold. Combine with `all: true` for 'every relevant doc above this confidence' agentic flows."
            },
            "all": {
                "type": "boolean",
                "default": false,
                "description": "Disable the top_k cap and return every hit that passed min_score (if set). Useful with output=\"files\" to enumerate every matching file in the corpus."
            },
            "output": {
                "type": "string",
                "enum": ["chunks", "files"],
                "default": "chunks",
                "description": "Output mode. 'chunks' returns one hit per chunk with snippet + line. 'files' dedupes by path and returns one hit per file (line=0, no snippet) — pairs well with `all: true` for listing every matching file."
            },
            "and": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Intersection terms. A chunk must score for the primary `query` AND every entry here to remain a result; final score is the harmonic mean of normalized per-query scores. Use to narrow a too-broad search. Example: query=\"authentication\", and=[\"rate limiting\"] → chunks about both."
            },
            "not": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Exclusion terms. Chunks scoring highly for any entry are dropped; weaker matches are demoted. Use to filter out a known distractor. Example: query=\"caching\", not=[\"Redis\"] → caching discussions that aren't primarily about Redis."
            }
        },
        "required": ["query"]
    });
    schema
        .as_object()
        .cloned()
        .expect("schema is object literal")
}

fn log_ready(state: &MultiSourceState) {
    eprintln!("dora mcp: ready with {} source(s):", state.order.len());
    for name in &state.order {
        let s = state.sources.get(name).expect("name in order map exists");
        eprintln!(
            "  - {} ({}, model={})",
            s.name,
            s.path.display(),
            s.embedder.id(),
        );
    }
}

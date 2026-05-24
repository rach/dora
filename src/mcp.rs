//! Stdio MCP server exposing two tools: `search` and `list_sources`.
//!
//! Multi-source by construction. A registry of N sources becomes a HashMap<name, SourceState>.
//! The single-source case (`dora mcp --source <path>`) is just a registry-of-one — same code path.
//!
//! Embedders are cached by `(provider, model, dimensions)` during boot: sources that share
//! the same model share a single Arc'd Embedder instance, saving ~80 MB per duplicate.

use anyhow::{Context, Result};
use rmcp::{
    ErrorData, RoleServer, ServerHandler, ServiceExt,
    model::{
        CallToolRequestParams, CallToolResult, Content, Implementation, ListToolsResult,
        PaginatedRequestParams, ServerCapabilities, ServerInfo, Tool, ToolAnnotations,
    },
    service::{MaybeSendFuture, RequestContext},
    transport::stdio,
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
    /// Maximum number of hits to return. Default 10, capped at 50.
    #[serde(default)]
    top_k: Option<u32>,
    /// Path-prefix filter applied within each searched source (e.g. "Daily/").
    #[serde(default)]
    path_prefix: Option<String>,
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

struct MultiSourceState {
    sources: HashMap<String, SourceState>,
    /// Stable ordered list of source names — matches insertion order for deterministic
    /// `list_sources` output and predictable cross-source iteration.
    order: Vec<String>,
}

#[derive(Clone)]
struct DoraServer {
    state: Arc<Mutex<MultiSourceState>>,
}

impl ServerHandler for DoraServer {
    fn get_info(&self) -> ServerInfo {
        let instructions = self
            .state
            .lock()
            .ok()
            .map(|s| build_instructions(&s));

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
    ) -> impl std::future::Future<Output = Result<ListToolsResult, ErrorData>> + MaybeSendFuture + '_ {
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

            Ok(ListToolsResult {
                tools: vec![search, list, find_def, find_callers, find_impls, repo_map],
                ..Default::default()
            })
        }
    }

    fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<CallToolResult, ErrorData>> + MaybeSendFuture + '_ {
        let state = self.state.clone();
        async move {
            match request.name.as_ref() {
                "search" => handle_search(&state, request).await,
                "list_sources" => handle_list_sources(&state).await,
                "find_definition" => handle_find_definition(&state, request).await,
                "find_callers" => handle_find_callers(&state, request).await,
                "find_implementations" => handle_find_implementations(&state, request).await,
                "repo_map" => handle_repo_map(&state, request).await,
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
    let path_prefix = args.path_prefix.as_deref();

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
            search_one(s, &args.query, top_k, path_prefix)
                .map_err(|e| ErrorData::internal_error(format!("search failed: {e}"), None))?
        }
        None => search_cross(multi, &args.query, top_k, path_prefix)
            .map_err(|e| ErrorData::internal_error(format!("cross-source search failed: {e}"), None))?,
    };

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
    serde_json::from_value(v).map_err(|e| ErrorData::invalid_params(format!("bad arguments: {e}"), None))
}

async fn handle_find_definition(
    state: &Arc<Mutex<MultiSourceState>>,
    request: CallToolRequestParams,
) -> Result<CallToolResult, ErrorData> {
    let args: FindDefinitionArgs = parse_args(request)?;
    let limit = args.limit.unwrap_or(10).clamp(1, 50) as usize;
    let symbol = args.symbol.trim().to_string();
    if symbol.is_empty() {
        return Err(ErrorData::invalid_params("symbol must not be empty".to_string(), None));
    }

    let mut guard = state
        .lock()
        .map_err(|e| ErrorData::internal_error(format!("state lock poisoned: {e}"), None))?;

    let names = match &args.source {
        Some(n) => {
            if !guard.sources.contains_key(n) {
                return Err(ErrorData::invalid_params(
                    format!("unknown source '{n}'. registered: {}", guard.order.join(", ")),
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
        return Err(ErrorData::invalid_params("symbol must not be empty".to_string(), None));
    }

    let mut guard = state
        .lock()
        .map_err(|e| ErrorData::internal_error(format!("state lock poisoned: {e}"), None))?;

    let names = match &args.source {
        Some(n) => {
            if !guard.sources.contains_key(n) {
                return Err(ErrorData::invalid_params(
                    format!("unknown source '{n}'. registered: {}", guard.order.join(", ")),
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
        return Err(ErrorData::invalid_params("symbol must not be empty".to_string(), None));
    }

    let mut guard = state
        .lock()
        .map_err(|e| ErrorData::internal_error(format!("state lock poisoned: {e}"), None))?;

    let names = match &args.source {
        Some(n) => {
            if !guard.sources.contains_key(n) {
                return Err(ErrorData::invalid_params(
                    format!("unknown source '{n}'. registered: {}", guard.order.join(", ")),
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
            format!("unknown source '{}'. registered: {}", args.source, guard.order.join(", ")),
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

    let entries = s.store.definitions_in_files(&file_ids).map_err(|e| {
        ErrorData::internal_error(format!("definitions_in_files: {e}"), None)
    })?;
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

fn search_one(
    s: &mut SourceState,
    query: &str,
    top_k: usize,
    path_prefix: Option<&str>,
) -> Result<Vec<crate::search::Hit>> {
    crate::search_with_self_heal(
        &s.path,
        &s.name,
        &s.cfg,
        &mut s.store,
        &s.chunker,
        s.embedder.as_ref(),
        query,
        top_k,
        path_prefix,
    )
}

/// Cross-source: over-fetch per source (2× top_k), then re-sort the merged list. RRF scores
/// are on the same scale (1/(60+rank)) so direct comparison is defensible.
fn search_cross(
    multi: &mut MultiSourceState,
    query: &str,
    top_k: usize,
    path_prefix: Option<&str>,
) -> Result<Vec<crate::search::Hit>> {
    let per_source_top = (top_k * 2).max(top_k);
    let mut all: Vec<crate::search::Hit> = Vec::new();
    for name in multi.order.clone() {
        let s = multi.sources.get_mut(&name).expect("name in order");
        match search_one(s, query, per_source_top, path_prefix) {
            Ok(hits) => all.extend(hits),
            Err(e) => {
                // Don't fail the whole call because one source errored — log + continue.
                eprintln!("dora mcp: source '{name}' search failed: {e}");
            }
        }
    }
    all.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    all.truncate(top_k);
    Ok(all)
}

// ---------------- entrypoints ----------------

/// Multi-source MCP server. Loads every source in the registry, sharing embedder instances
/// across sources that pick the same `(provider, model, dimensions)`.
pub async fn run_multi(registry: Registry) -> Result<()> {
    let mut state = build_state(&registry.sources)?;
    if state.sources.is_empty() {
        anyhow::bail!("no sources could be loaded — check logs above for per-source errors");
    }
    log_ready(&state);
    let server = DoraServer {
        state: Arc::new(Mutex::new(std::mem::take(&mut state))),
    };
    server
        .serve(stdio())
        .await
        .context("serve stdio MCP transport")?
        .waiting()
        .await
        .context("waiting on MCP service")?;
    Ok(())
}

/// Single-source MCP server (used by `dora mcp --source <path>`). Builds a synthetic
/// one-entry registry and reuses `run_multi`'s implementation.
pub async fn run(source_root: &Path) -> Result<()> {
    let source_root = source_root.canonicalize().context("canonicalize source path")?;
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
    run_multi(registry).await
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
    })
}

fn try_load_source(
    src: &Source,
    cache: &mut HashMap<String, DynEmbedder>,
) -> Result<SourceState> {
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

impl Default for MultiSourceState {
    fn default() -> Self {
        Self {
            sources: HashMap::new(),
            order: Vec::new(),
        }
    }
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
        let s = state
            .sources
            .get(name)
            .expect("name in order map exists");
        eprintln!(
            "  - {} ({}, model={})",
            s.name,
            s.path.display(),
            s.embedder.id(),
        );
    }
}

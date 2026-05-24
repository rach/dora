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

use crate::chunk::Chunker;
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
    chunker: Chunker,
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

            Ok(ListToolsResult {
                tools: vec![search, list],
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
    let chunker = Chunker::from_config(&cfg.chunking);

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

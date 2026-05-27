# Changelog

All notable changes to dora are documented here. Format roughly follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[SemVer](https://semver.org/spec/v2.0.0.html).

## [0.6.1] — 2026-05-27

### Changed

- **Default embedder swapped from `bge-small-en-v1.5` to
  `bge-base-en-v1.5-onnx-q`** (Qdrant's int8-quantized BGE-base). On the
  maintainer's 53-query eval fixture: R@1 lifts from **0.868 → 0.943**
  (+7.5 pts), MRR from **0.928 → 0.965**, and hard-paraphrase queries hit a
  perfect 1.000. Disk footprint is essentially the same (~33 MB), and CLI
  cold start is actually slightly faster (~0.15 s vs ~0.30 s for bge-small).
- **`mode = code` is unchanged**: Jina embeddings v2 base code stays the
  default for code sources.

### Validation

Head-to-head on 20 docs / 53 queries (mixed easy / medium / hard, paraphrase-heavy):

| embedder | size | R@1 | MRR | hard R@1 | CLI cold |
|---|---|---|---|---|---|
| `bge-small-en-v1.5` *(prior default)* | ~30 MB | 0.868 | 0.928 | 0.900 | ~0.30s |
| **`bge-base-en-v1.5-onnx-q`** *(new default)* | ~33 MB | **0.943** | **0.965** | **1.000** | **~0.15s** |
| `bge-base-en-v1.5` (full precision) | ~110 MB | 0.925 | 0.956 | 0.950 | ~0.33s |
| `embeddinggemma-300m-onnx` | ~333 MB | 0.943 | 0.967 | 0.950 | ~2s first |
| qmd-query (LLM-augmented stack) | — | 0.962 | 0.981 | 0.950 | ~0.65s |

dora's hybrid (bge-base-Q + FTS + literal + PRF) closes the gap to qmd's full
LLM-augmented pipeline to **1.9 pts on R@1** with no LLM dependency, and ties
the larger embeddinggemma model on R@1 at one-tenth the size.

### Migration note

Existing indexes carry the old `embedder_id` (`fastembed:Xenova/bge-small-en-v1.5`).
On the first `dora index` after upgrade, dora's existing `meta_matches` path
detects the mismatch, **wipes the source's DB, and re-embeds from scratch**.
Expect ~4–10s on a small notes folder, ~5 minutes on a 10k-file vault.

Users who want to keep the lighter bge-small embedder can pin it explicitly
per source:

```toml
# <source>/.dora/config.toml
[embedder]
provider = "fastembed"
model = "bge-small-en-v1.5"
```

### Docs

- **README — new "Choosing an embedder" subsection** under "What's happening
  under the hood", with the eval table above and a config-snippet showing
  how to opt into bge-small / embeddinggemma / OpenAI.
- **`docs/prds/optional-ollama-llm.md` permanently parked** — the 1.9 pt gap
  to qmd-query doesn't justify the ollama dependency.
- **`docs/ROADMAP.md` v0.8 (candle migration) demoted to "deferred"** —
  fastembed 5 already unlocks the embedder catalog (incl. embeddinggemma)
  we'd have used candle for.

## [0.6.0] — 2026-05-27

### Added

- **Pseudo-relevance feedback (PRF) as a fourth retrieval arm.** After FTS, vector
  ANN, and literal substring scan, dora now mines the top vector-ANN chunks for
  their most-frequent non-stopword, non-query-word tokens (up to 5) and runs them
  as an additional FTS query. The result joins the RRF merge alongside the other
  three arms, closing vocabulary gaps without an LLM. Always-on; no config knob.
- **Usage signal logging** via new `usage` table (migration #2). Every search
  records `(query_text, query_embedding, returned_chunks, used_chunk_id, ts)`.
  Data-collection only in v0.6 — v0.7 turns it into ranking signal, v0.9 turns it
  into LoRA fine-tuning input.
- **MCP-side use attribution.** When an MCP client calls `multi_get` on a path
  the most recent `search` (within 60s) returned, the matching `usage` row's
  `used_chunk_id` gets patched. Ring buffer caps at 64 recent searches; CLI
  invocations stay unattributed (best-effort signal).

### Validation

Head-to-head against qmd's published 18-query eval set (`test/eval-docs/`):

| config       | N  | R@1   | R@5   | R@10  | MRR   |
| ------------ | -- | ----- | ----- | ----- | ----- |
| dora (PRF)   | 18 | 1.000 | 1.000 | 1.000 | 1.000 |
| dora (no PRF)| 18 | 1.000 | 1.000 | 1.000 | 1.000 |
| ripgrep -l   | 18 |   .056|   .056|   .056|   .056|

dora's hybrid (FTS + vector + literal + PRF) hits perfect R@1 on every easy /
medium / hard query in qmd's fixture — no LLM, no Python, no ollama. qmd cannot
exceed 1.000 either, so we're within the PRD's "±5 R@5 points of qmd" acceptance
window. PRF lands clean (no regression vs. no-PRF on this fixture), but the
6-doc set is too small to differentiate; PRF's value will surface on diverse
real-world corpora where multiple docs compete for the same query. The eval
harness lives at `scripts/eval.sh` (maintainer-only, gitignored) for re-running
on broader fixtures as v0.7+ lands.

- **Minimalist colored CLI renderer** with inline previews. `dora "query"`
  now prints one header line per hit (`path:line  [heading]  ★score`, with
  bold-magenta / green / cyan / dim-yellow when stdout is a TTY) followed by
  up to 4 lines of preview from the matched chunk under a thin `│` bar.
  Pipes and `NO_COLOR=1` strip the ANSI codes automatically; `--json` and
  `--files` modes are unchanged.
- **First-time setup UX**. When a source uses an embedder model that isn't
  yet cached, dora prints `first-time setup — downloading embedder model ...`
  before fastembed's progress bar so users see immediate feedback instead of
  a blank terminal during the HF endpoint resolve.
- **README retrieval-pipeline diagram** (mermaid, renders inline on GitHub).
  Visualizes the four-arm fusion + per-file collapse + usage-table side-channel.

### Changed

- `Hit` now exposes `chunk_id` — needed for the MCP use-attribution path and the
  v0.7 signal-based reranker. JSON-serialized output gains the field.
- **`fastembed = "5"`** (was 4). Unlocks `embeddinggemma-300m-onnx` and a wider
  catalog without touching dora's embedder API. `TextEmbedding::embed` became
  `&mut self` in 5.x, so `FastembedEmbedder` wraps it in a Mutex to keep the
  `Embedder` trait's `&self` contract.

## [0.5.0] — 2026-05-27

### Added

- **`dora mcp --http`** — serve MCP over JSON-RPC over HTTP at `127.0.0.1:8181/mcp`
  (override with `--bind` / `--port`). One persistent server, models stay resident
  across requests, all MCP clients share. Closes the "every Claude Code launch
  reloads 200 MB of ONNX" problem.
- **`dora mcp --http --daemon`** — fork into the background, write PID to
  `~/.config/dora/mcp-http.pid`, log to a file next to the PID. Uses the
  `daemonize` crate (Unix-only). `keep_alive` semantics via brew-services.
- **`dora mcp stop`** — SIGTERM the daemon, escalate to SIGKILL after 5s, clean
  up the PID file.
- **`dora mcp status`** — GET `http://127.0.0.1:8181/health`, print uptime +
  registered sources, exit 0 if running, 1 if not.
- **`GET /health`** endpoint on the HTTP server returning
  `{status, version, uptime_secs, sources}` — used by `dora doctor` and
  `dora install` for transport auto-detection.
- **`dora install` transport auto-detection.** Before patching each client config,
  the installer checks whether the HTTP daemon is alive (PID + `/health`). If so,
  writes `{"url": "http://127.0.0.1:8181/mcp"}` into the MCP host config instead
  of the stdio launch command. If the daemon stops later, doctor surfaces the
  drift.
- **`dora doctor` MCP DAEMON section** — reports running/stopped state + uptime,
  cleans up stale PID files automatically.

### Changed

- **Brew service now runs `dora mcp --http`, not `dora watch`** (Homebrew formulas
  only support one `service` block). v0.2.1–v0.4.x users with
  `brew services start dora` running their watch process need to restart watch
  manually after upgrade — e.g. `nohup dora watch > /tmp/dora-watch.log 2>&1 &`,
  or create their own launchd plist. The HTTP daemon delivers a bigger UX win
  (no more cold-start ONNX reloads on every Claude Code launch) so the brew
  service slot goes to it by default.
- `mcp::run_multi` now takes a `Transport` (stdio or http) instead of being
  stdio-only. Existing callers updated; behavior unchanged when transport is
  stdio.
- `src/main.rs::cmd_mcp` reshaped: `dora mcp` keeps its stdio default; new flags
  `--http / --bind / --port / --daemon` and subcommands `stop / status` opt into
  the HTTP path.
- `tokio` features extended to `rt-multi-thread`, `signal`, `net` for the axum
  HTTP server. Stdio still uses the current-thread runtime; only HTTP spins up
  multi-thread.

### Notes

- HTTP daemon is **localhost-only by default** (`127.0.0.1`). Passing
  `--bind 0.0.0.0` is accepted but prints a warning — indexed content becomes
  searchable by anyone on the network.
- No TLS, no auth. For cross-machine setups, put a reverse proxy in front.
- Idle unload of per-source `Store` connections deferred to v0.5.1; the first
  cut keeps everything resident, predictable RAM cost.

## [0.4.1] — 2026-05-26

### Changed

- Per-subpath context strings now also surface in the **default text output** of
  `dora "<query>"`, not just `--json` / MCP. Each hit whose path is under a
  registered context prefix gets an indented continuation line:
  ```
  technology/Foo.md:1: [section] snippet…
         context: Engineering and design nuggets
  ```
  Closes a gap where v0.4.0's docs claimed context surfaces "on every matching
  hit" but the bare-CLI text path was dropping it. JSON + MCP behavior unchanged.

## [0.4.0] — 2026-05-26

### Added

- **`--min-score <f>`** CLI flag + MCP `search` arg. Drops hits below an RRF score
  threshold. Pair with `--all` for "every relevant document above this confidence"
  agentic flows.
- **`--all`** CLI flag + MCP `search` arg. Disables the top_k cap; returns every hit
  that passed `min_score` (if set).
- **`--files`** CLI flag + MCP `search` arg `output: "files"`. Dedupes hits by path,
  returns one entry per file (no `:line:` prefix, no snippet). Pairs with `--all`
  to enumerate every matching file. Pattern from qmd.
- **`multi_get` MCP tool**. Batch-retrieve documents by glob pattern relative to a
  registered source's root (`src/**/*.rs`, `notes/2026-*.md`). Returns body text
  per match, truncated at `max_bytes` (default 102400). Saves agents from N×Read
  round-trips when they already know which files they want.
- **`dora context <add|list|remove>`** + per-subpath context strings. Descriptive
  metadata attached to a path prefix within a source, surfaced as `Hit.context` on
  every matching search result. Use `/` as the prefix for source-wide default;
  subtree prefixes override the global one (longest-match wins with `/` boundary
  safety, so `/foo` doesn't match `/foobar`). New `contexts` table introduced via
  migration #1.
- **Forward-only DB migrations** (`src/migrations.rs`). Each `.dora/index.db` gains
  a `migrations(version, applied_at)` table; `Store::open` applies any new entries
  from the `MIGRATIONS` const slice idempotently. Sledgehammer (`SCHEMA_VERSION`)
  reserved for breaking changes that genuinely need re-embedding; additive changes
  ship as migrations and don't burn embedder time. v0.4 adds migration #1 (contexts);
  v0.5+ additive changes append from here.

### Changed

- **RRF merge gains a top-rank bonus** (+0.05 for rank-1 in any sub-arm, +0.02 for
  ranks 2-3) matching qmd's published fusion. Sharpens precision at the top of
  ranked results without affecting recall.
- `search::search()` signature now takes a `SearchOptions` struct instead of
  loose `top_k` + `path_prefix` args, so the new flags thread through cleanly.

## [0.3.1] — 2026-05-25

### Added

- **`codex` mode**: index OpenAI Codex CLI session transcripts under
  `~/.codex/sessions/YYYY/MM/DD/rollout-<iso>-<uuid>.jsonl`. Peer to
  `claude-code` — separate mode because Codex's JSONL schema differs
  (envelope `{timestamp, type, payload}`, split `function_call` /
  `function_call_output` records linked by `call_id`, `reasoning` blocks).
- Per-user-turn chunking matches the claude-code shape: one chunk = one user
  prompt + every assistant text / tool call / tool result until the next
  user prompt. Rendered as readable prose
  (`USER: …` / `ASSISTANT: …` / `[tool: exec_command ls -la]` /
  `[tool result exec_command: total 808 …]`).
- `heading_path = "<project> · <iso-minute> · codex"` — the trailing `codex`
  tag distinguishes Codex hits from Claude Code hits (`· branch:<branch>`)
  in cross-source search results.
- Auto-detection: paths ending in `.codex/sessions` resolve to `Mode::Codex`
  without needing `--mode`. `dora source add ~/.codex/sessions` defaults
  the source name to `codex`.
- `[codex] include_reasoning = false` (default, mirrors
  `[claude_code] include_thinking`); `settle_seconds = 60`.
- Codex `function_call_output` strips the standard
  `Chunk ID: … \nWall time: … \nProcess exited … \nOutput:\n` metadata
  header so the indexed text is just the real tool output.

### Changed

- Generalized the active-session settle filter in `run_incremental_index`
  from `mode == ClaudeCode` to `mode.is_transcript()`, which now covers both
  `Mode::ClaudeCode` and `Mode::Codex` (and any future agent transcript modes
  added to that helper). Same default-source-name pattern extended too.

## [0.3.0] — 2026-05-25

### Added

- **`claude-code` mode**: index Claude Code session transcripts under
  `~/.claude/projects/<encoded-cwd>/<session>.jsonl`. Each user-turn (one user
  prompt plus the assistant text + tool calls until the next user prompt)
  becomes one chunk, rendered as readable prose (`USER: …` / `ASSISTANT: …` /
  `[tool: …]` / `[tool result: …]`). `heading_path` carries
  `<project> · <iso-minute> · branch:<git-branch>` so search results are
  project-anchored — no ugly `-Users-rachid-Dev-…` encoded paths surface.
- Auto-detection: paths ending in `.claude/projects` resolve to `claude-code`
  mode without needing `--mode`. `dora source add` defaults the source name
  to `claude-code` in that case (the basename `"projects"` would be useless).
- Settle filter: JSONL files whose mtime is newer than
  `[claude_code] settle_seconds = 60` are skipped (they're being written to
  by the active session; re-embedding every flush burns the embedder). Skipped
  count appears in the index summary as `N settling`.
- `[claude_code] include_thinking = false` (default): assistant `thinking`
  blocks are excluded from chunk bodies — they bloat embeddings without
  improving recall. Opt back in via the config field if you want them indexed.
- One-liner setup: `dora source add ~/.claude/projects` (auto-detects + names
  the source `claude-code`).

### Notes

- Other agent transcripts (Codex, Aider, Cursor) would get their own mode +
  chunker — JSONL shapes differ per tool. The pattern is established by this
  release.
- No secret-redaction filter. Embedding vectors aren't reversible to
  plaintext; the FTS5 + chunks tables store the same text already on disk in
  `~/.claude/projects/`. If a use case for redaction emerges, that's a
  separate v0.x feature.

## [0.2.6] — 2026-05-25

### Changed

- FTS5 now indexes each chunk's `heading_path` alongside the body. The
  markdown chunker strips heading lines from chunk content (the heading
  is kept as metadata), which left the BM25 arm of RRF blind to queries
  that match a section title verbatim — the embedding arm already saw
  the heading via `chunk::embedded_text`, but FTS only saw the body.
  This closes that gap.
- Measured on `test-corpora/rust-book/src` against a 77-query set mined
  from real H2/H3 headings (titles that don't appear verbatim in the
  body, so heading signal is the only handle): R@1 0.571 → 0.740,
  MRR 0.734 → 0.860, R@5 0.961 → 1.000. 21 queries improved, 0
  regressed, 56 unchanged.
- Existing installs need to reindex to pick up the change — old FTS
  rows still contain body-only text. `rm -rf <source>/.dora/index.db &&
  dora index <source>` on each registered source.

## [0.2.5] — 2026-05-25

### Changed

- `find` wrapper picks up the same path-aware shape grep/rg/ag got in v0.2.4.
  Two intercept shapes now (both flagless, both require the phrase arg to
  contain whitespace so single-token queries stay in `grep`):
  - `find "natural phrase"` — PWD must resolve into a dora source.
  - `find <dir> "natural phrase"` — `<dir>` must resolve into a dora source.
  Anything else (flags like `-name`/`-type`/`-newer`, multiple paths,
  unquoted phrases) falls through to real `find` unchanged.
- Existing installs run `dora install` once to pick up the new template.

## [0.2.4] — 2026-05-25

### Changed

- The shell wrapper for `grep` / `rg` / `ag` no longer falls through on
  every flag. It now intercepts the common-search shape:
  `grep -r "pattern" [path...]` (and any combination of `-r -R -i -n -H`
  flags) — exactly the form people actually type. Behavior:
  - Allowed flag chars: `r R i n H`. Any other flag → real grep.
  - Pattern arg + optional directory path args. If no path given, PWD is
    used. Each path is resolved to absolute and walked up looking for
    `.dora/index.db`; if every path lands inside a registered dora
    source, the wrapper cd's into the first source root in a subshell
    and runs `dora "$pattern"`. Otherwise real grep.
  - File paths (vs directories) fall through — the user clearly wants
    real grep on a specific file.
- Net effect: `grep -r "Reciprocal Rank Fusion" .` from inside a notes
  vault now returns dora's semantic hits instead of running real grep
  recursively. Same for `grep -r "..." ~/Dev/myproject` from anywhere.
  `grep -E`, `grep -F`, `grep -v`, `grep -l`, etc. still fall through.
- Existing installs need to run `dora install` once to pick up the new
  wrapper template (the dora-managed block in `.zshrc` is rewritten
  idempotently).

## [0.2.3] — 2026-05-25

### Added

- `dora wrappers <on|off|status>` subcommand toggles the shell wrappers
  (`grep`/`rg`/`ag`/`find` injected by `dora install`) without editing
  `.zshrc`. State persists in a new global config file
  `~/.config/dora/config.toml` under `[wrappers] enabled`. Default: enabled.
- `dora wrappers status -q` exits 0 if enabled, 1 if disabled — used by the
  wrapper template itself as the hot-path check.
- Doctor's `~/.zshrc` line now surfaces the active toggle state ("active" vs
  "installed but disabled").

### Changed

- Wrapper template in `~/.zshrc` calls `dora wrappers status -q 2>/dev/null`
  at the top of each function. If dora is missing/broken or the toggle's off,
  the wrapper falls through to the real `grep`/`rg`/`ag`/`find`. Existing
  installs need to run `dora install` once to pick up the new template
  (idempotent rewrite of the dora-managed block).

## [0.2.2] — 2026-05-24

### Changed

- Hybrid search gains a third arm: a literal-substring `LIKE` scan over
  `chunks.content`, merged into the existing FTS5 + vector RRF. Closes the gaps
  where FTS5's tokenization can't help — camelCase identifiers
  (`processRequest`), snake_case adjacency (`foo_bar`), magic constants
  (`E_NOENT`, `MAX_RETRY_COUNT`), short error codes. Always on; the per-query
  cost is invisible on personal-vault sizes.
- `dora "upsert_file_with_chunks"` and similar literal-identifier queries now
  return the right chunk directly, without falling through to `rg`. The skill's
  anti-pattern carve-out for literal queries shrank accordingly.
- Refactored `rrf_merge(fts, ann)` → `rrf_merge_n(&[…])` so future arms drop
  in without further plumbing.

## [0.2.1] — 2026-05-24

### Added

- `brew services start dora` support. Tap formula gains a `service do … end` block
  (`run [opt_bin/"dora", "watch"]`, `keep_alive true`, logs to
  `/opt/homebrew/var/log/dora-watch.log`). Users on Homebrew can manage the watcher
  with `brew services {start|stop|restart|list} dora` and it auto-restarts on login
  and on crash.

### Changed

- `dora watch` no longer bails when the registry is empty. Instead, it logs
  `no sources registered yet — waiting` and proceeds into the main loop; the
  notify-on-registry hook (added in v0.2.0) picks up the first `dora source add`
  whenever it lands. Required for the `brew services start dora` flow on a fresh
  install — users can `brew install` + `brew services start dora` before
  registering any source without launchd respawn-looping a failing process.

## [0.2.0] — 2026-05-24

### Added

- **Code-aware sources via tree-sitter.** Six languages on day 1: Rust, Python,
  TypeScript + JavaScript, Go, Java, Ruby. Each grammar's `queries/tags.scm`
  drives chunking — one chunk per function / method / class / module / etc.
  — and emits reference edges into a new `links` table.
- **Mode system.** `dora source add --mode <obsidian|notes|docs|code|auto>`
  picks a complete preset (chunker, embedder, file extensions, ignore-dirs).
  `auto` (the default) detects by `.obsidian/` presence and code-vs-md file
  ratio. Resolved mode is persisted as `[source] mode = "..."` in each
  source's `.dora/config.toml`.
- **`links` table + two-pass resolver.** Pass 1 resolves within-file edges
  immediately (`confidence='exact'`). Pass 2 resolves cross-file edges via
  symbol-name match — `exact` if unique, `name_match` if ambiguous.
- **PageRank scoring** (Aider-style). Identifier-quality weighting
  (real-name × 10, private × 0.1, generic-name × 0.1) and `focus_paths`
  personalization (× 50) so the agent can bias ranking toward what it's
  currently editing.
- **Four new MCP tools:**
  - `find_definition(symbol, source?)` — locate where a symbol is defined.
  - `find_callers(symbol, source?, depth?)` — recursive CTE over the call
    graph, max depth 5. Each result carries a `confidence` field.
  - `find_implementations(symbol, source?)` — trait / interface implementors.
  - `repo_map(source, focus_paths?, token_budget?)` — PageRank-ranked outline
    of the codebase, rendered to fit a token budget.
- **`dora source add --mode`** flag with auto-detect summary at registration time.
- **Doctor** now reports per-source `mode` and (for code sources) a chunk-kind
  breakdown + link count.
- Code-mode embedder default switched to `jina-embeddings-v2-base-code`
  (markdown modes stay on `bge-small-en-v1.5`).
- Ruby joins the code mode language registry (six languages total: Rust,
  Python, TypeScript+JavaScript, Go, Java, Ruby).
- Bundled `dora` Claude skill under `skills/dora/SKILL.md`. Auto-loads in any
  folder with `.dora/index.db`; mode-aware playbook routes Claude to dora's
  MCP tools (`find_definition`, `find_callers`, `find_implementations`,
  `repo_map`, `search`) instead of grep. Installable via `npx skills add
  rach/dora`, the Claude Code `/plugin` marketplace
  (`/plugin marketplace add github.com/rach/dora` then
  `/plugin install dora@dora`), or manual symlink. See [skills/README.md](skills/README.md).
- Plugin marketplace manifests at `.claude-plugin/{marketplace,plugin}.json`
  so the dora repo doubles as a single-plugin Claude Code marketplace.
- Homebrew tap at [rach/homebrew-dora](https://github.com/rach/homebrew-dora).
  Install via `brew install rach/dora/dora`. The macOS Apple Silicon bottle
  resolves to the v0.2.0 GitHub release tarball; Intel macOS + Linux users
  are pointed at `cargo install --git https://github.com/rach/dora --tag v0.2.0`
  pending cross-platform bottle support.

### Changed

- `Chunker` is now a trait (`src/chunk/mod.rs`) with two impls:
  `markdown::MarkdownChunker` (existing behavior, unchanged) and
  `code::CodeChunker` (new). Schema bumped to `3`, chunker version to `3`.
- `vault::list_entries` accepts an explicit `allow_exts` list — the walker
  no longer hard-codes `.md`. Mode-aware extension allow-list is computed
  at `run_incremental_index` entry.
- Config file (`.dora/config.toml`) gained a `[source]` section. Two-layer
  resolution: raw TOML → mode defaults → user overrides → final `Config`.
  Every section remains optional; most users only set `[source] mode`.
- `chunks` table gained `kind`, `symbol`, `parent_chunk_id` columns with
  indices on `kind` and `symbol`.

### Notes

- Markdown sources from v0.1 keep working with identical retrieval results.
  Bumping `schema_version` + `chunker_version` triggers a clean rebuild on
  next `dora index`; the rebuild produces the same embeddings since the
  embedded-text contract for markdown is preserved.
- LSP integration is deferred indefinitely. Tree-sitter alone covers
  `find_definition` / `find_callers` / `find_implementations` / `repo_map`
  with the `confidence` field surfacing ambiguity honestly.

## [0.1.0] — 2026-05

Initial release.

- CLI: `dora index`, `dora "query"`, `dora source <add|list|remove|describe>`,
  `dora install`, `dora doctor`, `dora mcp`, `dora watch`.
- MCP server with `search` and `list_sources` tools, multi-source registry.
- Hybrid retrieval: FTS5 + sqlite-vec ANN merged via Reciprocal Rank Fusion.
- Adaptive markdown chunker with frontmatter-prose synthesis for spec-sheet
  notes.
- Local-first embeddings via fastembed (~25 model catalog) + OpenAI provider.
- Shell wrappers (`grep` / `rg` / `ag` / `find`) hijack flagless calls inside
  registered folders.
- macOS Apple Silicon prebuilt binary.

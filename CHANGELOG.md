# Changelog

All notable changes to dora are documented here. Format roughly follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[SemVer](https://semver.org/spec/v2.0.0.html).

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

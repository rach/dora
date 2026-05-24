---
name: dora
description: >
  Help the user navigate, search, and reason about code repos AND notes/vault folders that are
  registered with dora (https://github.com/rach/dora). Replaces grep-then-read loops with direct
  calls to dora's MCP tools (search, find_definition, find_callers, find_implementations,
  repo_map, list_sources). Use when the user asks where a symbol is defined, who calls a function,
  what implements a trait/interface, asks for an overview or outline of a codebase, or asks any
  "how does this work / where does X happen" question in a code repo. ALSO use when the user asks
  "what did I write about X", "do I have anything on Y", or any retrieval question against
  personal notes, journal entries, design docs, or an Obsidian vault. Activates whenever the
  current folder contains .dora/index.db or is registered via `dora source list`.
---

# dora — semantic + structural search for code and notes

dora is a single-binary local search index. Once a folder is registered (`dora source add`), it
exposes 6 MCP tools that Claude should prefer over grep/Glob/Read for the question types listed
in the description above. This skill tells you which tool to reach for, in which order, based on
what *mode* the source is in.

## Step 1 — detect the source mode

Before answering, find out whether you're working in a **code** source or a **notes** source.
This decides the rest of the playbook.

Options, in order of cost:

1. Read `.dora/config.toml` in the current folder. Look for `[source] mode = "..."`.
2. Call `mcp__dora__list_sources()` and find the entry whose `path` matches the current folder.
   The response includes the embedder id — `jinaai/jina-embeddings-v2-base-code` means code mode,
   `Xenova/bge-small-en-v1.5` means notes/obsidian/docs mode.
3. Run `dora source list` in the shell and check the output.

If no `.dora/index.db` exists in the folder and the path isn't registered, **dora isn't set up
for this folder**. Tell the user and fall back to grep/Glob/Read. Don't pretend dora is available.

## Step 2 — code playbook (`mode = code`)

Pick the tool by question shape:

| User asked... | Use this | Notes |
|---|---|---|
| "where is `X` defined?" | `mcp__dora__find_definition({symbol: "X", source})` | Never grep when you know the symbol name. |
| "what calls `X`?" / "what breaks if I change `X`?" | `mcp__dora__find_callers({symbol: "X", source, depth: 1})` | Each result has `confidence: "exact" \| "name_match"`. `name_match` = candidate (multiple symbols share the name) — verify before reporting. Bump `depth` (max 5) for transitive walks. |
| "who implements `Trait`?" / "find all classes implementing `Interface`" | `mcp__dora__find_implementations({symbol: "Trait", source})` | Covers Rust `impl Trait for X`, Java/TS `implements`. |
| "find code that does X" (conceptual, no exact name) | `mcp__dora__search({query: "...", source})` | Semantic search. Use this to surface candidates, then `find_definition` on the names that come back. |
| "give me an outline of this repo" / "what's important here?" | `mcp__dora__repo_map({source, focus_paths: [<files the user is editing>], token_budget: 2000})` | PageRank-ranked outline. `focus_paths` biases the ranking toward neighbors of the files you're already in — pass paths the user just opened/edited. |

Always pass `source` (the registered source name) so dora knows which index to hit. If you don't
know the name, call `mcp__dora__list_sources()` once at the start.

## Step 3 — notes playbook (`mode = obsidian` / `notes` / `docs`)

Notes mode uses a smaller toolset — `search` does most of the work.

- **"Did I write about X?" / "do I have notes on Y?"** → `mcp__dora__search({query: "X", source})`.
  Run this *before* answering "no" or "I don't see anything" — embeddings find notes by meaning,
  not exact words. The user's past phrasing is almost certainly different from their current.
- **"I want to start a note on Y"** → search first, surface any existing notes so the user
  doesn't duplicate. Suggest extending the existing note when there's a hit.
- **Cross-source recall** ("what did I decide about caching, and where is it implemented?") →
  call `mcp__dora__search({query: "..."})` *without* a `source` arg. dora merges across every
  registered source — vault + code — and ranks by score.
- **Frontmatter-only notes** (kepano spec sheets: movies, books, places, contacts) look empty in
  grep but dora synthesizes the YAML into prose. They ARE searchable — don't tell the user
  "nothing matched" if you only checked the body.

## Step 4 — anti-patterns (when NOT to use dora)

dora is for *semantic* and *structural* queries. As of v0.2.2 it also handles literal-substring matching natively (camelCase, snake_case, magic constants, error codes), so most "exact string" queries belong in `search` too. Reach for the built-in tools instead when:

- **File-path patterns** (`*.test.ts`, `2026-05-*.md`, "every file in docs/") → `Glob`.
- **`#obsidian-tag` searches** → `Grep` is exact and faster.
- **Bulk literal scans over a folder dora hasn't indexed** → `Grep` / `rg`. dora only sees what's been registered + indexed.
- **Tiny repos (<100 files)** where a `Grep` will scan in <100ms — no setup needed.
- **Reading a specific file you already know about** → `Read`.

It's totally fine to combine: `mcp__dora__search` to find the relevant chunks, then `Read` for
full file context, then `Edit`. Don't force every step through dora.

## Tool reference

All six tools are exposed as `mcp__dora__<name>`:

- `search(query, source?, top_k=10, path_prefix?)` — hybrid FTS5 + vector ANN, merged via
  Reciprocal Rank Fusion. Works in any mode. Omit `source` for cross-source search.
- `list_sources()` — every registered source with name, description, path, embedder id, file +
  chunk counts. Call this once at the start of a session if you don't know the source name.
- `find_definition(symbol, source?, limit=10)` — definition chunks where `symbol == ?`. Code mode.
- `find_callers(symbol, source?, depth=1, limit=20)` — recursive CTE over the call graph, max
  depth 5. Each result carries `confidence`. Code mode.
- `find_implementations(symbol, source?, limit=20)` — chunks implementing a trait/interface. Code
  mode.
- `repo_map(source, focus_paths=[], token_budget=2000)` — PageRank-ranked outline. **Requires**
  `source` (scores aren't comparable across separately-computed graphs). Code mode.

## Quick troubleshooting

- **"no such source"** errors → run `mcp__dora__list_sources()` to see what's registered. The
  user may need to `dora source add <path>` first.
- **Empty results from `find_definition`** → the symbol might not be a definition (it's only a
  reference). Try `search` to find chunks that mention it.
- **`name_match` confidence in `find_callers`** → multiple symbols share the name. Open the
  candidates with `Read` to confirm which is the real caller before reporting impact.
- **Stale results** → dora self-heals on every search (re-walks if mtimes changed). If results
  still look stale, ask the user to run `dora index` or check `dora doctor`.

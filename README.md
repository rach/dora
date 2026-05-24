<div align="center">
  <img src="assets/dora.png" alt="dora — the file explorer" width="280">

  <h1>dora</h1>
  <p><em>the file explorer</em></p>
</div>

---

**dora turns a folder of markdown notes into something Claude (and Cursor / Codex) can search by meaning, not just keywords.**

You point dora at a directory — your Obsidian vault, work notes, project docs, anything with `.md` files. It builds a tiny local search index next to those files. Then:

- **From Claude Code**: ask *"what did I write about hook design?"* and Claude pulls the actual passage from your notes.
- **From your terminal**: type `grep "the bit I half-remember about ranking algorithms"` and get the right note ranked first.
- **As a tool you call directly**: `dora "any natural language query"` returns ranked hits with a one-line excerpt.

That's the whole product. One small binary. Local-only by default — no API key, no cloud, no daemon running in the background, no kernel-level filesystem trickery. If you want cloud embeddings (OpenAI) for better quality, that's a one-line config change.

## What it actually looks like

After a one-time setup:

```sh
$ cd ~/notes
$ grep "hooks matter more than the body"
x/Hooks matter more than content.md:1: The first line of any post does more work...
post/Houston you have a data problem.md:14: ...if your hook doesn't earn the second sentence...
```

That's not literal grep — that's *semantic* search. dora hijacks flagless `grep` (and `rg`, `ag`, `find`) when you're inside a registered notes folder. `grep -i pattern` and any flagged form still call real grep.

In Claude Code (after `dora install`):

```
You    > what did I decide about the embedder trait?
Claude > [calls mcp__dora__search]
       > You decided to defer extracting the trait until a real second impl
       > (OpenAI) was in scope, per smfs's lesson — quoted from:
       >   technology/Defer trait abstraction until the second impl.md
       >   "Single trait in code with one impl. We deliberately defer the
       >    trait abstraction until there's a second use case to motivate
       >    the shape — same lesson stash's BaseEmbedder teaches."
```

## Install

### macOS — Apple Silicon (recommended)

Download the prebuilt binary from the latest release:

```sh
# Download + install in one go
curl -L -o /tmp/dora https://github.com/rach/dora/releases/latest/download/dora-fs-v0.1.0-macos-arm64
chmod +x /tmp/dora
xattr -d com.apple.quarantine /tmp/dora 2>/dev/null   # bypass macOS Gatekeeper warning
sudo mv /tmp/dora /usr/local/bin/dora                  # or anywhere on your $PATH
dora --version                                          # should print: dora 0.1.0
```

> **macOS Gatekeeper note**: the binary is unsigned. The `xattr` line removes the quarantine attribute so Gatekeeper doesn't warn. If you skip it, the first launch will say *"cannot be opened because the developer cannot be verified"* — fix with: right-click the binary in Finder → Open → "Open Anyway." Or run the `xattr` line above.

### Intel Mac / Linux / latest `main`

Build from source. You need Rust installed (`brew install rust` or [rustup.rs](https://rustup.rs/)).

```sh
git clone git@github.com:rach/dora.git
cd dora
cargo build --release
sudo cp target/release/dora /usr/local/bin/dora
dora --version
```

(Cross-platform prebuilt binaries via GitHub Actions are planned but not yet shipped.)

## First run (5 minutes)

```sh
# 1. Index a folder of notes
dora index ~/path/to/your/notes

# 2. Register it in the global registry
dora source add ~/path/to/your/notes

# 3. Patch Claude Code (+ Cursor + Codex) MCP configs + install shell wrappers
dora install

# 4. Verify everything's wired up
dora doctor
```

Then restart Claude Code (or Cursor / Codex). It'll see a `mcp__dora__search` tool automatically.

Try a query from the terminal:

```sh
cd ~/path/to/your/notes
dora "what did I write about X?"
# or, after `dora install`, with the grep wrapper:
grep "what did I write about X?"
```

You can register more folders any time:

```sh
dora index ~/work/notes
dora source add ~/work/notes --name work \
  --description "Work meeting notes + design docs"

dora index ~/code-snippets
dora source add ~/code-snippets --name snippets

dora source list
```

## What's happening under the hood

```
~/your-notes/
├── note.md                ← you write these
├── deep/folder/...
└── .dora/                 ← dora writes here only (gitignorable)
    ├── index.db           ← local SQLite database with the search index
    └── models/            ← downloaded ML model (~80 MB, one time)

~/.config/dora/
└── registry.toml          ← list of folders you've registered
```

**Indexing.** dora reads each `.md` file, splits it into chunks (respecting headings, code blocks, tables), generates a vector embedding per chunk using a small local ML model (default: BGE-small, ~80 MB ONNX file that runs on your laptop). Stores everything in SQLite.

**Searching.** When you query, dora embeds the query the same way, then does two searches in parallel: a keyword-based one (BM25 / FTS5) and a vector-similarity one. It merges the two ranked lists with a technique called Reciprocal Rank Fusion. You get the top N results back.

**Incremental.** After the first index, only changed files get re-embedded. Detected via mtime + content hash. Renames are detected and don't re-embed. Even on a vault with 2,500+ chunks (e.g. the Rust Book), a no-op re-index takes about 130 milliseconds.

**Self-healing.** When you query, dora notices if any files changed since the last walk and quietly catches up before searching. So results are always fresh, even if you forgot to re-index.

**Multi-folder.** All registered folders are searchable from a single MCP server (one process, one model in memory). Claude can scope a search to one folder by name, or search across everything and merge results.

## Commands

```
dora index [<path>]                                    # build/update the index for <path> (defaults to cwd)
dora "<query>" [--top-k N] [--json]                    # search the index in cwd
dora source add <path> [--name N] [--description "…"]  # register a folder globally
dora source list                                       # show registered folders
dora source remove <name>                              # unregister
dora source describe <name> "…"                        # update a folder's description
dora install [--client …] [--include …] [--wrap …]     # patch MCP host configs + shell wrappers
dora doctor                                            # health check, exit code reflects status
dora mcp [--include …] [--exclude …] [--source <path>] # run the MCP server (usually called by Claude Code itself)
dora watch [--include …] [--exclude …]                 # foreground watcher that keeps things fresh proactively
```

Every command has a `--help`. Full reference (with examples for each command, config file format, embedding-model choices, and Claude Code wiring details) is below.

## Full reference

### `dora index [<path>] [--dry-run]`

Incremental index. First run does everything; subsequent runs only re-embed changed files. Atomic per-file. `--dry-run` walks + chunks but doesn't embed; useful with OpenAI to preview cost.

```sh
dora index                         # cwd
dora index ~/notes                 # specific folder
dora index --dry-run               # show what would happen + estimated cost
```

### `dora "query" [--top-k N] [--json]`

Hybrid search against the index in cwd. Self-heals (runs a diff first if files have changed).

```sh
dora "what did I decide about X?"
dora --top-k 3 "Rust lifetimes"
dora --json "query" | jq .          # machine-readable
```

### `dora source <add|list|remove|describe>`

Manages the global registry at `~/.config/dora/registry.toml`. Folders here become searchable from a single `dora mcp` server.

```sh
dora source add ~/notes
dora source add ~/work --name work --description "Work meeting notes + design docs"
dora source describe brain "Personal design + engineering notes"
dora source list
dora source remove work
```

Descriptions are shown to Claude — they help it pick the right folder when you have several registered.

### `dora install [--client …] [--include …] [--wrap …] [--no-shell]`

Idempotent. Patches your MCP host configs (Claude Code, Cursor, Codex) with a `dora` MCP server entry, and adds zsh wrappers for `grep`, `rg`, `ag`, `find` to `~/.zshrc`. Skips clients whose config doesn't exist.

```sh
dora install                                   # default: all detected clients + all 4 wrappers
dora install --client claude                   # only Claude Code
dora install --include work,docs               # scope this MCP entry to a subset of registered folders
dora install --wrap grep                       # only the grep wrapper (removes the others if present)
dora install --no-shell                        # skip wrappers entirely
```

The wrappers are inert if the underlying tool isn't installed — `command rg` just errors out same as if our wrapper wasn't there. Safe to install them all by default.

### `dora doctor`

Single-screen health check:

```
BINARY
  ✓ /Users/me/.../dora
  ✓ version 0.0.1
REGISTRY
  ✓ brain     /Users/me/notes, walked 22m ago
  ⚠ stale     last walked 14d ago — `dora index` to refresh
MCP HOSTS
  ✓ Claude Code   `dora` registered
  ⚠ Cursor        present but no `dora` entry — `dora install --client cursor`
SHELL
  ✓ dora wrappers: grep, rg, ag, find
WATCHER
  ✓ dora watch running (pid 12345)
Result: 0 errors, 1 warning
```

Exits 1 if anything is broken. Run it after `dora install` or whenever something feels off.

### `dora watch [--include …] [--exclude …]`

Foreground watcher that keeps registered folders fresh proactively. Debounces bursts of edits (500ms) and only re-embeds files that actually changed. Ctrl-C to stop. Optional — search auto-heals without watch.

```sh
dora watch                                # watch every registered folder
dora watch --include brain                # watch just one
nohup dora watch > /tmp/dora-watch.log &   # background it
```

### `dora mcp [--include …] [--exclude …] [--source <path>]`

The MCP server, normally launched by Claude Code itself (via the config `dora install` patched in). Exposes two tools:

- `mcp__dora__search(query, source?, top_k?, path_prefix?)` — hybrid search, optionally scoped to one registered folder.
- `mcp__dora__list_sources()` — list registered folders with descriptions + counts.

Concurrent processes are safe (WAL mode, per-file transactions, 5s busy timeout). Embedders are shared across folders that use the same model — three folders on the default model load the ONNX file once, not three times.

```sh
dora mcp                                  # serve all registered folders
dora mcp --include brain,work             # subset
dora mcp --source ~/some/ad-hoc/folder    # ad-hoc, ignoring the registry
```

### Wrapped tools

After `dora install`, inside any folder with a `.dora/index.db`:

```sh
grep "natural language query"            # → semantic search via dora
rg "concurrent state design"             # → semantic search via dora
ag "system architecture"                 # → semantic search via dora
find "what did I write about hooks"      # → semantic search via dora (single quoted phrase only)

# These always fall through to the real tool:
grep -F "literal string"
grep -ri "case insensitive"
rg --files
find . -name "*.md" -mtime -7
```

`find`'s wrapper is conservative — it only intercepts a single arg containing whitespace, because find's normal forms (`find . -name x`) need to stay untouched.

## Configuration (optional)

Per-folder `.dora/config.toml` — every key is optional with sensible defaults:

```toml
[chunking]
target_bytes        = 1800     # chunk size target; ~450 tokens
atomic_below_bytes  = 1600     # files smaller than this stay as one chunk
overlap_bytes       = 270      # ~15% overlap on recursive paragraph splits

[embedder]
provider    = "fastembed"      # or "openai"
model       = "bge-small-en-v1.5"
api_key_env = "OPENAI_API_KEY" # only for provider = "openai"
# dimensions = 1024            # openai-only

[search]
top_k             = 10
collapse_per_file = true       # at most one hit per file
```

Switching `model` triggers a clean re-index on next run.

**Local embedding models** (any of ~25 from fastembed's catalog): `bge-small-en-v1.5` (default), `bge-base-en-v1.5`, `bge-large-en-v1.5`, `multilingual-e5-small/base/large`, `all-minilm-l6-v2`, `nomic-embed-text-v1.5`, `jina-embeddings-v2-base-code`, `mxbai-embed-large-v1`, `modernbert-embed-large`, `bge-small-zh-v1.5`, `bge-large-zh-v1.5`, and more. Pick by name in config.

**OpenAI**: `text-embedding-3-small` (default 1536d, supports custom dims), `text-embedding-3-large` (3072d), `text-embedding-ada-002` (legacy). `dora index --dry-run` prints estimated cost before any API call.

## Project status

**Working end-to-end** — CLI, MCP, install/doctor, watcher, multi-source registry, four shell wrappers, OpenAI integration. Used daily against multiple personal vaults.

**Not yet release-grade** — no automated tests, no Homebrew formula, prebuilt binary is Apple Silicon only. To try it on Intel Mac or Linux, clone + `cargo build --release` yourself.

## License

MIT — see [LICENSE](LICENSE).

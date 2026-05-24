# dora skill

A bundled Claude skill that teaches Claude Code (and any other agent that reads
`SKILL.md` files) *when* to reach for dora's MCP tools instead of grep/Glob/Read.

Without this skill, an agent pointed at a dora-indexed folder will often default
to `Grep` — exactly the failure mode dora was built to fix. With it installed,
Claude routes "where is `X` defined?" → `mcp__dora__find_definition`, "what did
I write about Y?" → `mcp__dora__search`, "give me an outline of this repo" →
`mcp__dora__repo_map`, and so on.

One skill, mode-aware. The body of [`dora/SKILL.md`](dora/SKILL.md) branches on
the source's resolved mode (`code` vs `obsidian` / `notes` / `docs`) — same
switch dora itself uses.

## Prerequisites

You need dora installed and wired into Claude Code first. From the dora README:

```sh
# 1. install the binary
curl -L -o /tmp/dora https://github.com/rach/dora/releases/latest/download/dora-fs-v0.2.0-macos-arm64
chmod +x /tmp/dora && xattr -d com.apple.quarantine /tmp/dora 2>/dev/null
sudo mv /tmp/dora /usr/local/bin/dora

# 2. register at least one source
dora index ~/your-folder
dora source add ~/your-folder

# 3. wire the MCP server into Claude Code
dora install
```

If `dora doctor` shows everything green, you're ready to install the skill.

## Install — three paths

### Path 1 (recommended): `npx skills`

Cross-agent, one command. Works with Claude Code, OpenCode, Cursor, and
anything else that reads SKILL.md.

```sh
npx skills add rach/dora
```

`npx skills` auto-discovers every `SKILL.md` under this repo's `skills/`
directory and installs it into your active agent's skills location (usually
`~/.claude/skills/`). Re-run the command to update. Pass `--list` to see what's
in the repo before installing:

```sh
npx skills add rach/dora --list
```

### Path 2: Claude Code plugin (`/plugin marketplace`)

Native to Claude Code. Integrates with the `/plugin` slash-command UI.

```
/plugin marketplace add github.com/rach/dora
/plugin install dora@dora
```

The first command registers this repo as a plugin marketplace (using the
`.claude-plugin/marketplace.json` manifest at the repo root). The second
installs the `dora` plugin from it.

- **Update**: `/plugin marketplace update dora` then `/plugin update dora@dora`.
- **Uninstall**: `/plugin uninstall dora@dora`.
- **Remove the marketplace too**: `/plugin marketplace remove dora`.

### Path 3 (fallback): manual git clone + symlink

Always-works baseline. Useful on older Claude Code versions or any other agent
without a package-manager-style installer.

```sh
git clone https://github.com/rach/dora ~/Dev/dora
mkdir -p ~/.claude/skills
ln -s ~/Dev/dora/skills/dora ~/.claude/skills/dora
```

- **Update**: `cd ~/Dev/dora && git pull`. The symlink picks up the new content.
- **Uninstall**: `rm ~/.claude/skills/dora`.

### Per-project install

Any of the three paths can install into a specific project instead of globally,
by targeting that project's `.claude/skills/` directory. For example, for the
manual path:

```sh
cd ~/Dev/myproject
ln -s ~/Dev/dora/skills/dora .claude/skills/dora
```

The skill then only activates while that project is the active workspace.

## Verify

Restart Claude Code. In a dora-registered folder, ask a question the skill
should route through dora:

- In a **code source**: *"where is `upsert_file_with_chunks` defined?"* → Claude
  should auto-call `mcp__dora__find_definition` (visible in the tool-call
  trace) rather than running `Grep`.
- In a **notes vault**: *"what did I write about Reciprocal Rank Fusion?"* →
  Claude should auto-call `mcp__dora__search`.

If the same prompt previously triggered `Grep` and now triggers an `mcp__dora__*`
call, the skill is steering tool choice as intended.

## Troubleshooting

**Skill not activating** — confirm it's installed where the agent expects:

| Install path | Check |
|---|---|
| `npx skills` | `ls ~/.claude/skills/` should include `dora` |
| `/plugin install` | `/plugin list` should show `dora@dora` enabled |
| manual symlink | `ls -la ~/.claude/skills/dora` should show a symlink (or directory) containing `SKILL.md` |

**Skill installed but Claude still greps** — the skill's `description` field
needs to overlap with what Claude reads in the user's prompt. If your phrasing
is very different from the trigger language ("find that function I wrote about
auth" instead of "where is `validateAuth` defined?"), Claude may not match. Add
"dora" by name to your prompt to force activation while you're testing
(`"use dora to find where validateAuth is defined"`).

**`mcp__dora__*` tools aren't visible at all** — the skill is independent of
the MCP server. If the tools don't appear in `/mcp` output, the server isn't
wired up. Run `dora install --client claude` and restart Claude Code.

**`dora doctor` flags warnings** — fix those first; the skill assumes a healthy
dora install.

## What this skill does NOT do

- It does not install dora itself, register sources, or wire up the MCP server.
  Those are one-time setup steps documented in the main [README](../README.md).
- It does not add slash commands like `/dora-search`. Auto-loading on
  description match is the cleaner UX; explicit commands would compete with the
  MCP tools that should already be there.
- It does not work without the dora binary + MCP server. The skill's whole job
  is to route Claude to the MCP tools — without them it has nothing to route to.

## Adding more skills later

The `skills/` directory is plural by design. Future additions (e.g. a
watch-tuning skill, a doctor-debugging skill) drop into `skills/<name>/SKILL.md`
and become installable through all three paths above without further work.

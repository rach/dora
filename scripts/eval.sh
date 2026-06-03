#!/usr/bin/env bash
# Committed dora eval gate.
#
# This is the contributor-facing quality check for retrieval changes. It only runs
# committed fixtures through dora's debug-only `eval` command; qmd comparisons are
# historical/manual context, not part of the maintained gate.

set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== cargo test =="
cargo test

echo
echo "== notes regression guard =="
cargo run -- eval fixtures/eval/notes --min-r-at-1 0.9

echo
echo "== code regression guard =="
cargo run -- eval fixtures/eval/code --min-r-at-1 0.8

echo
echo "== linked graph gate =="
cargo run -- eval fixtures/eval/linked --compare-disable-graph

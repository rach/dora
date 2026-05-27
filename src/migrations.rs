//! Forward-only DB migrations applied to each source's `.dora/index.db` on `Store::open`.
//!
//! Distinct from `SCHEMA_VERSION` in `config.rs`: that's the sledgehammer for breaking
//! changes that require re-embedding (chunker semantics, embedder swap, etc.) — when it
//! mismatches, `cmd_index` wipes the DB and rebuilds. This module handles the additive
//! changes (new tables, new indexes) that don't touch embeddings; the loop applies any
//! migration whose version is greater than what's recorded in the `migrations` table.
//!
//! **Append-only**: never reorder, never delete, never edit an already-applied migration —
//! you'd corrupt existing users' DBs. Always add new migrations at the end with the next
//! integer version. Each migration runs in its own transaction; on failure, the transaction
//! rolls back and `Store::open` propagates the error, so the next attempt retries cleanly.
//!
//! When you need a change that DOES require re-embedding, bump `SCHEMA_VERSION` instead.

use anyhow::Result;
use rusqlite::{params, Connection};

/// Ordered list of `(version, SQL)`. New entries go at the end with the next version.
pub const MIGRATIONS: &[(i64, &str)] = &[
    // v0.4: per-subpath context strings surfaced alongside search hits.
    (
        1,
        "CREATE TABLE IF NOT EXISTS contexts ( \
             id           INTEGER PRIMARY KEY, \
             path_prefix  TEXT NOT NULL UNIQUE, \
             description  TEXT NOT NULL, \
             updated_at   INTEGER NOT NULL \
         ); \
         CREATE INDEX IF NOT EXISTS idx_contexts_prefix ON contexts(path_prefix);",
    ),
    // v0.6: usage signal. Logged best-effort on every search; the `used_chunk_id` is
    // patched in if a follow-up MCP `multi_get` (or future Read-equivalent) reads a path
    // the search just returned. Feeds v0.7's signal-based reranker and v0.9's LoRA pass.
    (
        2,
        "CREATE TABLE IF NOT EXISTS usage ( \
             id              INTEGER PRIMARY KEY, \
             query_text      TEXT NOT NULL, \
             query_embedding BLOB NOT NULL, \
             returned_chunks TEXT NOT NULL, \
             used_chunk_id   INTEGER, \
             created_at      INTEGER NOT NULL \
         ); \
         CREATE INDEX IF NOT EXISTS idx_usage_created ON usage(created_at);",
    ),
];

/// Apply any migrations whose version is newer than the `MAX(version)` in the
/// `migrations` table. Creates the table itself if it doesn't exist yet.
pub fn run(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS migrations ( \
             version    INTEGER PRIMARY KEY, \
             applied_at INTEGER NOT NULL \
         );",
    )?;
    let current: i64 = conn.query_row(
        "SELECT COALESCE(MAX(version), 0) FROM migrations",
        [],
        |r| r.get(0),
    )?;
    for (version, sql) in MIGRATIONS.iter() {
        if *version <= current {
            continue;
        }
        let tx = conn.unchecked_transaction()?;
        tx.execute_batch(sql)?;
        tx.execute(
            "INSERT INTO migrations (version, applied_at) VALUES (?, ?)",
            params![*version, now_secs()],
        )?;
        tx.commit()?;
    }
    Ok(())
}

fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

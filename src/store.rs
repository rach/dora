//! SQLite + sqlite-vec persistence.
//!
//! v0 item B grows the `files` table with `mtime`/`size`/`content_hash` so incremental indexing
//! can skip unchanged files. WAL mode is enabled so concurrent `dora "query"` doesn't block a
//! long `dora index` (load-bearing for v0 item E `dora watch`).

use anyhow::{Context, Result};
use rusqlite::{params, params_from_iter, Connection};
use std::collections::HashMap;
use std::path::Path;

pub fn init_sqlite_vec() {
    use rusqlite::ffi::sqlite3_auto_extension;
    use sqlite_vec::sqlite3_vec_init;
    unsafe {
        sqlite3_auto_extension(Some(std::mem::transmute(
            sqlite3_vec_init as *const (),
        )));
    }
}

pub struct Store {
    conn: Connection,
}

pub struct ChunkRow<'a> {
    pub idx: usize,
    pub heading_path: &'a str,
    pub content: &'a str,
    pub start_byte: usize,
    pub end_byte: usize,
    pub embedding: &'a [f32],
    /// Semantic kind ("prose" / "function" / "method" / "class" / etc.). Stored as text so
    /// new kinds can be added without a schema migration.
    pub kind: &'a str,
    /// Symbol name for code chunks (function/struct/class name). None for prose.
    pub symbol: Option<&'a str>,
    /// Index (within this file's chunk slice) of the enclosing chunk, if any. Resolved to a
    /// DB id during insert because chunks are inserted parent-first.
    pub parent_chunk_idx: Option<usize>,
}

/// A pre-resolution edge handed to the store. `target_chunk_id` is `None` at insert time
/// for cross-file edges; pass-2 (`resolve_cross_file_links`) fills them in by symbol match.
pub struct LinkRow<'a> {
    /// Chunk-idx within the file (resolved to DB id during insert).
    pub source_chunk_idx: usize,
    pub kind: &'a str,
    pub target_symbol: &'a str,
    pub target_path: Option<&'a str>,
}

/// Snapshot of a `files` row, used by the diff loop to decide insert/update/touch/skip.
#[derive(Debug, Clone)]
pub struct FileRow {
    pub id: i64,
    pub mtime: u64,
    pub size: u64,
    pub content_hash: String,
}

pub struct FetchedChunk {
    pub heading_path: String,
    pub content: String,
    pub start_byte: usize,
    pub path: String,
}

#[derive(Debug, Clone)]
pub struct DefinitionHit {
    pub chunk_id: i64,
    pub heading_path: String,
    pub content: String,
    pub start_byte: usize,
    pub kind: String,
    pub symbol: String,
    pub path: String,
}

#[derive(Debug, Clone)]
pub struct CallerHit {
    pub chunk_id: i64,
    pub heading_path: String,
    pub content: String,
    pub start_byte: usize,
    pub kind: String,
    pub symbol: String,
    pub path: String,
    pub distance: usize,
    pub confidence: String,
}

#[derive(Debug, Clone)]
pub struct OutlineEntry {
    pub path: String,
    pub file_id: i64,
    pub symbol: String,
    pub kind: String,
    pub heading_path: String,
    pub start_byte: usize,
}

impl Store {
    pub fn open(path: &Path, embed_dims: usize) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let conn = Connection::open(path).context("open sqlite db")?;
        // WAL + NORMAL synchronous: readers never block writers, writers serialize naturally.
        // busy_timeout makes us safe under multi-process contention (e.g., two `dora mcp`
        // servers both touching the same source, or a `dora index` running while a server
        // is serving). 5s is generous for any realistic personal-vault workload.
        // Set before any other writes.
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        conn.busy_timeout(std::time::Duration::from_millis(5000))?;
        let s = Self { conn };
        s.create_schema(embed_dims)?;
        Ok(s)
    }

    fn create_schema(&self, embed_dims: usize) -> Result<()> {
        let sql = format!(
            r#"
            CREATE TABLE IF NOT EXISTS files (
                id           INTEGER PRIMARY KEY,
                path         TEXT UNIQUE NOT NULL,
                mtime        INTEGER NOT NULL DEFAULT 0,
                size         INTEGER NOT NULL DEFAULT 0,
                content_hash TEXT    NOT NULL DEFAULT '',
                indexed_at   INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id              INTEGER PRIMARY KEY,
                file_id         INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                chunk_idx       INTEGER NOT NULL,
                heading_path    TEXT,
                content         TEXT NOT NULL,
                start_byte      INTEGER NOT NULL,
                end_byte        INTEGER NOT NULL,
                kind            TEXT NOT NULL DEFAULT 'prose',
                symbol          TEXT,
                parent_chunk_id INTEGER REFERENCES chunks(id) ON DELETE SET NULL,
                UNIQUE(file_id, chunk_idx)
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_symbol ON chunks(symbol) WHERE symbol IS NOT NULL;
            CREATE INDEX IF NOT EXISTS idx_chunks_kind   ON chunks(kind);
            CREATE TABLE IF NOT EXISTS links (
                id              INTEGER PRIMARY KEY,
                source_chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                kind            TEXT NOT NULL,
                target_chunk_id INTEGER REFERENCES chunks(id) ON DELETE SET NULL,
                target_symbol   TEXT,
                target_path     TEXT,
                confidence      TEXT NOT NULL DEFAULT 'exact'
            );
            CREATE INDEX IF NOT EXISTS idx_links_source     ON links(source_chunk_id);
            CREATE INDEX IF NOT EXISTS idx_links_target_id  ON links(target_chunk_id);
            CREATE INDEX IF NOT EXISTS idx_links_target_sym ON links(target_symbol);
            CREATE INDEX IF NOT EXISTS idx_links_kind       ON links(kind);
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                content, content='chunks', content_rowid='id'
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
                chunk_id INTEGER PRIMARY KEY, embedding FLOAT[{embed_dims}]
            );
            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            "#
        );
        self.conn.execute_batch(&sql)?;
        Ok(())
    }

    // ---------- meta ----------

    pub fn get_meta(&self, key: &str) -> Result<Option<String>> {
        let v = self
            .conn
            .query_row("SELECT value FROM meta WHERE key = ?", params![key], |r| {
                r.get::<_, String>(0)
            });
        match v {
            Ok(s) => Ok(Some(s)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    pub fn set_meta(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) \
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![key, value],
        )?;
        Ok(())
    }

    // ---------- diff helpers ----------

    /// Snapshot every `files` row keyed by relative path. The diff loop in main joins this
    /// against `vault::list_entries` output to classify each entry as
    /// Insert/Update/Touch/Skip/Delete.
    pub fn list_files(&self) -> Result<HashMap<String, FileRow>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, path, mtime, size, content_hash FROM files")?;
        let rows = stmt.query_map([], |row| {
            let id: i64 = row.get(0)?;
            let path: String = row.get(1)?;
            let mtime: i64 = row.get(2)?;
            let size: i64 = row.get(3)?;
            let content_hash: String = row.get(4)?;
            Ok((
                path,
                FileRow {
                    id,
                    mtime: mtime as u64,
                    size: size as u64,
                    content_hash,
                },
            ))
        })?;
        let mut out = HashMap::new();
        for r in rows {
            let (path, row) = r?;
            out.insert(path, row);
        }
        Ok(out)
    }

    /// mtime-only change: file content unchanged, just update the recorded mtime so the
    /// stat-only short-circuit fires next run.
    pub fn touch_file_mtime(&self, path: &str, mtime: u64) -> Result<()> {
        self.conn.execute(
            "UPDATE files SET mtime = ?, indexed_at = ? WHERE path = ?",
            params![mtime as i64, now_secs(), path],
        )?;
        Ok(())
    }

    pub fn delete_file(&self, path: &str) -> Result<()> {
        // FK ON DELETE CASCADE handles chunks; the chunks DELETE triggers FTS contentless
        // cleanup automatically. chunks_vec is owned by file_id-less chunk_id rows, so we
        // collect those first.
        let chunk_ids: Vec<i64> = {
            let mut stmt = self.conn.prepare(
                "SELECT c.id FROM chunks c JOIN files f ON f.id = c.file_id WHERE f.path = ?",
            )?;
            let rows = stmt.query_map(params![path], |row| row.get::<_, i64>(0))?;
            let mut out = Vec::new();
            for r in rows {
                out.push(r?);
            }
            out
        };
        let tx = self.conn.unchecked_transaction()?;
        for cid in &chunk_ids {
            tx.execute("DELETE FROM chunks_vec WHERE chunk_id = ?", params![cid])?;
            tx.execute("DELETE FROM chunks_fts WHERE rowid = ?", params![cid])?;
        }
        tx.execute("DELETE FROM files WHERE path = ?", params![path])?;
        tx.commit()?;
        Ok(())
    }

    /// Detected rename: same content_hash, different path. UPDATE in place — no embed cost.
    pub fn rename_file(&self, old_path: &str, new_path: &str, mtime: u64, size: u64) -> Result<()> {
        self.conn.execute(
            "UPDATE files SET path = ?, mtime = ?, size = ?, indexed_at = ? WHERE path = ?",
            params![new_path, mtime as i64, size as i64, now_secs(), old_path],
        )?;
        Ok(())
    }

    /// Upsert: replaces existing row + chunks atomically. Used for both new files (Insert)
    /// and content-changed files (Update). Each call is one SQLite transaction.
    pub fn upsert_file_with_chunks(
        &mut self,
        path: &str,
        mtime: u64,
        size: u64,
        content_hash: &str,
        chunks: &[ChunkRow],
        links: &[LinkRow],
    ) -> Result<()> {
        let tx = self.conn.transaction()?;

        // Clear existing rows for this path (if any). Cascades drop chunks; we also drop
        // the FTS + vec rows we own keyed by chunk_id. links to/from those chunks cascade
        // (source CASCADE) or get NULL'd (target SET NULL — pass-2 will re-resolve).
        let existing_chunk_ids: Vec<i64> = {
            let mut stmt = tx.prepare(
                "SELECT c.id FROM chunks c JOIN files f ON f.id = c.file_id WHERE f.path = ?",
            )?;
            let rows = stmt.query_map(params![path], |row| row.get::<_, i64>(0))?;
            let mut out = Vec::new();
            for r in rows {
                out.push(r?);
            }
            out
        };
        for cid in &existing_chunk_ids {
            tx.execute("DELETE FROM chunks_vec WHERE chunk_id = ?", params![cid])?;
            tx.execute("DELETE FROM chunks_fts WHERE rowid = ?", params![cid])?;
        }
        tx.execute("DELETE FROM files WHERE path = ?", params![path])?;

        let now = now_secs();
        tx.execute(
            "INSERT INTO files (path, mtime, size, content_hash, indexed_at) \
             VALUES (?, ?, ?, ?, ?)",
            params![path, mtime as i64, size as i64, content_hash, now],
        )?;
        let file_id = tx.last_insert_rowid();

        // chunks are sorted parent-first by the chunker — we can resolve parent_chunk_id by
        // looking up the already-inserted parent's DB id at each insertion.
        let mut idx_to_id: HashMap<usize, i64> = HashMap::with_capacity(chunks.len());
        let mut symbol_to_id: HashMap<&str, i64> = HashMap::new();

        for c in chunks {
            let parent_id = c
                .parent_chunk_idx
                .and_then(|pidx| idx_to_id.get(&pidx).copied());
            tx.execute(
                "INSERT INTO chunks (file_id, chunk_idx, heading_path, content, start_byte, end_byte, kind, symbol, parent_chunk_id) \
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                params![
                    file_id,
                    c.idx as i64,
                    c.heading_path,
                    c.content,
                    c.start_byte as i64,
                    c.end_byte as i64,
                    c.kind,
                    c.symbol,
                    parent_id,
                ],
            )?;
            let chunk_id = tx.last_insert_rowid();
            idx_to_id.insert(c.idx, chunk_id);
            if let Some(sym) = c.symbol {
                symbol_to_id.insert(sym, chunk_id);
            }

            tx.execute(
                "INSERT INTO chunks_fts (rowid, content) VALUES (?, ?)",
                params![chunk_id, c.content],
            )?;

            let bytes = f32_vec_to_bytes(c.embedding);
            tx.execute(
                "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
                params![chunk_id, bytes],
            )?;
        }

        // Insert edges. Within-file matches (target_symbol equals one of this file's chunk
        // symbols) resolve immediately with confidence 'exact'. The rest are stored with
        // target_chunk_id NULL — `resolve_cross_file_links` patches them after the whole
        // index pass completes.
        for link in links {
            let Some(&source_id) = idx_to_id.get(&link.source_chunk_idx) else {
                continue;
            };
            let within_file = symbol_to_id.get(link.target_symbol).copied();
            tx.execute(
                "INSERT INTO links (source_chunk_id, kind, target_chunk_id, target_symbol, target_path, confidence) \
                 VALUES (?, ?, ?, ?, ?, ?)",
                params![
                    source_id,
                    link.kind,
                    within_file,
                    link.target_symbol,
                    link.target_path,
                    if within_file.is_some() { "exact" } else { "exact" },
                ],
            )?;
        }

        tx.commit()?;
        Ok(())
    }

    /// Pass-2 resolver. Fills in `target_chunk_id` and updates `confidence` for any links
    /// where the target wasn't found in the source chunk's own file. Runs as two UPDATEs
    /// keyed off `(target_symbol, target_path)`. Cheap even on large indexes — symbol is
    /// indexed and the planner can use it.
    pub fn resolve_cross_file_links(&mut self) -> Result<usize> {
        let tx = self.conn.transaction()?;
        // First, NULL out any target_chunk_id where the chunk it points to is gone — handles
        // the case where a file got re-indexed and inbound links still hold the old id.
        // (SET NULL on FK does this for hard deletes, but we run it explicitly to also catch
        //  links that were exact and now need re-checking after a same-symbol target moved.)
        // Step 1: for unresolved links, attempt unique-symbol resolution.
        tx.execute(
            "UPDATE links \
             SET target_chunk_id = ( \
               SELECT c.id FROM chunks c \
               WHERE c.symbol = links.target_symbol \
               LIMIT 1 \
             ), \
             confidence = ( \
               SELECT CASE WHEN COUNT(*) = 1 THEN 'exact' ELSE 'name_match' END \
               FROM chunks c WHERE c.symbol = links.target_symbol \
             ) \
             WHERE target_chunk_id IS NULL \
               AND target_symbol IS NOT NULL \
               AND target_symbol != ''",
            [],
        )?;
        // Count how many links resolved (have non-NULL target).
        let resolved: i64 = tx.query_row(
            "SELECT COUNT(*) FROM links WHERE target_chunk_id IS NOT NULL",
            [],
            |r| r.get(0),
        )?;
        tx.commit()?;
        Ok(resolved as usize)
    }

    pub fn count_links(&self) -> Result<i64> {
        Ok(self
            .conn
            .query_row("SELECT COUNT(*) FROM links", [], |r| r.get(0))?)
    }

    /// Read-only access for callers that need to run analytical queries (PageRank, MCP
    /// repo_map). Returns the underlying connection — kept narrow to avoid leaking write
    /// access from outside the Store.
    pub fn conn(&self) -> &Connection {
        &self.conn
    }

    // ---------- code-aware lookups (sub-slice E) ----------

    pub fn find_definitions(&self, symbol: &str, limit: usize) -> Result<Vec<DefinitionHit>> {
        let mut stmt = self.conn.prepare(
            "SELECT c.id, c.heading_path, c.content, c.start_byte, c.kind, c.symbol, f.path \
             FROM chunks c JOIN files f ON f.id = c.file_id \
             WHERE c.symbol = ? AND c.kind NOT IN ('prose') \
             LIMIT ?",
        )?;
        let rows = stmt.query_map(params![symbol, limit as i64], |row| {
            Ok(DefinitionHit {
                chunk_id: row.get(0)?,
                heading_path: row.get::<_, Option<String>>(1)?.unwrap_or_default(),
                content: row.get(2)?,
                start_byte: row.get::<_, i64>(3)? as usize,
                kind: row.get(4)?,
                symbol: row.get::<_, Option<String>>(5)?.unwrap_or_default(),
                path: row.get(6)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Direct callers (depth=1) and transitive callers (depth>1). Uses a recursive CTE so
    /// the call doesn't N+1 across depth levels.
    pub fn find_callers(
        &self,
        symbol: &str,
        depth: usize,
        limit: usize,
    ) -> Result<Vec<CallerHit>> {
        let depth = depth.clamp(1, 5);
        // The CTE walks: start at any chunk whose symbol matches → follow incoming `calls`
        // edges → expand up to `depth` hops. Each visited caller gets emitted with its
        // distance from the original target.
        let sql = "
            WITH RECURSIVE
              targets(id) AS (
                SELECT id FROM chunks WHERE symbol = ?1 AND kind NOT IN ('prose')
              ),
              callers(caller_id, target_id, dist, confidence) AS (
                SELECT l.source_chunk_id, l.target_chunk_id, 1, l.confidence \
                  FROM links l \
                  JOIN targets t ON t.id = l.target_chunk_id \
                  WHERE l.kind = 'calls'
                UNION
                SELECT l.source_chunk_id, l.target_chunk_id, c.dist + 1, l.confidence \
                  FROM links l \
                  JOIN callers c ON c.caller_id = l.target_chunk_id \
                  WHERE l.kind = 'calls' AND c.dist < ?2
              )
            SELECT c.id, c.heading_path, c.content, c.start_byte, c.kind, c.symbol, \
                   f.path, callers.dist, callers.confidence \
              FROM callers JOIN chunks c ON c.id = callers.caller_id \
                           JOIN files f ON f.id = c.file_id \
              ORDER BY callers.dist, f.path \
              LIMIT ?3";
        let mut stmt = self.conn.prepare(sql)?;
        let rows = stmt.query_map(params![symbol, depth as i64, limit as i64], |row| {
            Ok(CallerHit {
                chunk_id: row.get(0)?,
                heading_path: row.get::<_, Option<String>>(1)?.unwrap_or_default(),
                content: row.get(2)?,
                start_byte: row.get::<_, i64>(3)? as usize,
                kind: row.get(4)?,
                symbol: row.get::<_, Option<String>>(5)?.unwrap_or_default(),
                path: row.get(6)?,
                distance: row.get::<_, i64>(7)? as usize,
                confidence: row.get(8)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn find_implementations(
        &self,
        symbol: &str,
        limit: usize,
    ) -> Result<Vec<DefinitionHit>> {
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT c.id, c.heading_path, c.content, c.start_byte, c.kind, c.symbol, f.path \
             FROM links l \
             JOIN chunks c ON c.id = l.source_chunk_id \
             JOIN files f ON f.id = c.file_id \
             WHERE l.kind = 'implements' AND l.target_symbol = ? \
             LIMIT ?",
        )?;
        let rows = stmt.query_map(params![symbol, limit as i64], |row| {
            Ok(DefinitionHit {
                chunk_id: row.get(0)?,
                heading_path: row.get::<_, Option<String>>(1)?.unwrap_or_default(),
                content: row.get(2)?,
                start_byte: row.get::<_, i64>(3)? as usize,
                kind: row.get(4)?,
                symbol: row.get::<_, Option<String>>(5)?.unwrap_or_default(),
                path: row.get(6)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// All definition chunks for a list of files, ordered (file_id, chunk_idx). Used by
    /// `repo_map` to render an outline once PageRank has picked the top files.
    pub fn definitions_in_files(&self, file_ids: &[i64]) -> Result<Vec<OutlineEntry>> {
        if file_ids.is_empty() {
            return Ok(Vec::new());
        }
        let placeholders = std::iter::repeat("?")
            .take(file_ids.len())
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "SELECT f.path, f.id, c.symbol, c.kind, c.heading_path, c.start_byte \
             FROM chunks c JOIN files f ON f.id = c.file_id \
             WHERE c.file_id IN ({placeholders}) \
               AND c.symbol IS NOT NULL AND c.symbol != '' \
             ORDER BY f.path, c.chunk_idx"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(params_from_iter(file_ids), |row| {
            Ok(OutlineEntry {
                path: row.get(0)?,
                file_id: row.get(1)?,
                symbol: row.get::<_, Option<String>>(2)?.unwrap_or_default(),
                kind: row.get(3)?,
                heading_path: row.get::<_, Option<String>>(4)?.unwrap_or_default(),
                start_byte: row.get::<_, i64>(5)? as usize,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    // ---------- search ----------

    pub fn search_fts(
        &self,
        query: &str,
        limit: usize,
        path_prefix: Option<&str>,
    ) -> Result<Vec<i64>> {
        match path_prefix {
            None => {
                let mut stmt = self.conn.prepare(
                    "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? \
                     ORDER BY bm25(chunks_fts) LIMIT ?",
                )?;
                let rows = stmt
                    .query_map(params![query, limit as i64], |row| row.get::<_, i64>(0))?;
                let mut hits = Vec::new();
                for r in rows {
                    hits.push(r?);
                }
                Ok(hits)
            }
            Some(prefix) => {
                let like = format!("{}%", prefix);
                let mut stmt = self.conn.prepare(
                    "SELECT cf.rowid FROM chunks_fts cf \
                     JOIN chunks c ON c.id = cf.rowid \
                     JOIN files f ON f.id = c.file_id \
                     WHERE chunks_fts MATCH ? AND f.path LIKE ? \
                     ORDER BY bm25(chunks_fts) LIMIT ?",
                )?;
                let rows = stmt.query_map(params![query, like, limit as i64], |row| {
                    row.get::<_, i64>(0)
                })?;
                let mut hits = Vec::new();
                for r in rows {
                    hits.push(r?);
                }
                Ok(hits)
            }
        }
    }

    pub fn search_ann(
        &self,
        query_vec: &[f32],
        limit: usize,
        path_prefix: Option<&str>,
    ) -> Result<Vec<i64>> {
        let bytes = f32_vec_to_bytes(query_vec);
        match path_prefix {
            None => {
                let mut stmt = self.conn.prepare(
                    "SELECT chunk_id FROM chunks_vec WHERE embedding MATCH ? \
                     ORDER BY distance LIMIT ?",
                )?;
                let rows = stmt
                    .query_map(params![bytes, limit as i64], |row| row.get::<_, i64>(0))?;
                let mut hits = Vec::new();
                for r in rows {
                    hits.push(r?);
                }
                Ok(hits)
            }
            Some(prefix) => {
                // sqlite-vec's MATCH wants its own LIMIT inside the virtual table query, so we
                // over-fetch then filter by path in a wrapping query. The over-fetch factor (4×)
                // is a heuristic — keeps recall high without unbounded work.
                let like = format!("{}%", prefix);
                let inner_limit = (limit * 4).max(50) as i64;
                let mut stmt = self.conn.prepare(
                    "SELECT v.chunk_id FROM (\
                       SELECT chunk_id, distance FROM chunks_vec \
                       WHERE embedding MATCH ? ORDER BY distance LIMIT ?\
                     ) v \
                     JOIN chunks c ON c.id = v.chunk_id \
                     JOIN files f ON f.id = c.file_id \
                     WHERE f.path LIKE ? \
                     ORDER BY v.distance LIMIT ?",
                )?;
                let rows = stmt.query_map(
                    params![bytes, inner_limit, like, limit as i64],
                    |row| row.get::<_, i64>(0),
                )?;
                let mut hits = Vec::new();
                for r in rows {
                    hits.push(r?);
                }
                Ok(hits)
            }
        }
    }

    pub fn fetch_chunk(&self, chunk_id: i64) -> Result<Option<FetchedChunk>> {
        let mut stmt = self.conn.prepare(
            "SELECT c.heading_path, c.content, c.start_byte, f.path \
             FROM chunks c JOIN files f ON f.id = c.file_id WHERE c.id = ?",
        )?;
        let mut rows = stmt.query(params![chunk_id])?;
        if let Some(row) = rows.next()? {
            Ok(Some(FetchedChunk {
                heading_path: row.get::<_, Option<String>>(0)?.unwrap_or_default(),
                content: row.get(1)?,
                start_byte: row.get::<_, i64>(2)? as usize,
                path: row.get(3)?,
            }))
        } else {
            Ok(None)
        }
    }

    pub fn count_files(&self) -> Result<i64> {
        Ok(self
            .conn
            .query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0))?)
    }

    pub fn count_chunks(&self) -> Result<i64> {
        Ok(self
            .conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0))?)
    }

    pub fn file_ids_for_chunks(&self, ids: &[i64]) -> Result<HashMap<i64, i64>> {
        if ids.is_empty() {
            return Ok(HashMap::new());
        }
        let placeholders = std::iter::repeat("?")
            .take(ids.len())
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!("SELECT id, file_id FROM chunks WHERE id IN ({placeholders})");
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(params_from_iter(ids), |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?))
        })?;
        let mut out = HashMap::new();
        for r in rows {
            let (c, f) = r?;
            out.insert(c, f);
        }
        Ok(out)
    }
}

fn f32_vec_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for f in v {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

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
                id           INTEGER PRIMARY KEY,
                file_id      INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                chunk_idx    INTEGER NOT NULL,
                heading_path TEXT,
                content      TEXT NOT NULL,
                start_byte   INTEGER NOT NULL,
                end_byte     INTEGER NOT NULL,
                UNIQUE(file_id, chunk_idx)
            );
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
    ) -> Result<()> {
        let tx = self.conn.transaction()?;

        // Clear existing rows for this path (if any). Cascades drop chunks; we also drop
        // the FTS + vec rows we own keyed by chunk_id.
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

        for c in chunks {
            tx.execute(
                "INSERT INTO chunks (file_id, chunk_idx, heading_path, content, start_byte, end_byte) \
                 VALUES (?, ?, ?, ?, ?, ?)",
                params![
                    file_id,
                    c.idx as i64,
                    c.heading_path,
                    c.content,
                    c.start_byte as i64,
                    c.end_byte as i64,
                ],
            )?;
            let chunk_id = tx.last_insert_rowid();

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

        tx.commit()?;
        Ok(())
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

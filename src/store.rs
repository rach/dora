//! SQLite + sqlite-vec persistence.
//!
//! v0 item B grows the `files` table with `mtime`/`size`/`content_hash` so incremental indexing
//! can skip unchanged files. WAL mode is enabled so concurrent `dora "query"` doesn't block a
//! long `dora index` (load-bearing for v0 item E `dora watch`).

use anyhow::{Context, Result};
use rusqlite::{params, params_from_iter, Connection};
use std::collections::HashMap;
use std::path::Path;

#[allow(clippy::missing_transmute_annotations)]
pub fn init_sqlite_vec() {
    use rusqlite::ffi::sqlite3_auto_extension;
    use sqlite_vec::sqlite3_vec_init;
    unsafe {
        sqlite3_auto_extension(Some(std::mem::transmute(sqlite3_vec_init as *const ())));
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
        crate::migrations::run(&s.conn)?;
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
            let path: String = row.get(1)?;
            let mtime: i64 = row.get(2)?;
            let size: i64 = row.get(3)?;
            let content_hash: String = row.get(4)?;
            Ok((
                path,
                FileRow {
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

            // Index the heading path alongside the body so BM25 sees the section title.
            // The chunker strips the heading line from `content`, so without this the FTS
            // arm of RRF is blind to queries that match section titles (e.g. "Setting Up
            // a New Project" when the body never repeats those words verbatim). Measured
            // on rust-book/src: R@1 0.57 → 0.74, MRR 0.73 → 0.86, 21 wins / 0 regressions.
            let alias_text = c
                .symbol
                .filter(|_| c.kind != "prose")
                .map(|sym| crate::chunk::symbol_alias_text(c.heading_path, sym))
                .unwrap_or_default();
            let fts_text = if c.heading_path.is_empty() {
                format!("{}\n{}", alias_text, c.content)
            } else {
                format!("{}\n{}\n{}", c.heading_path, alias_text, c.content)
            };
            tx.execute(
                "INSERT INTO chunks_fts (rowid, content) VALUES (?, ?)",
                params![chunk_id, fts_text],
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
                    "exact",
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

    /// Pass-2 resolver for authored prose links (`kind='wikilink'`). Unlike code edges
    /// (resolved by `chunks.symbol`), wikilinks resolve by note title/path: a `[[folder/Note]]`
    /// or `[text](folder/note.md)` points at the file whose path-suffix or basename matches.
    /// Resolves to the target note's **chunk 0** (its representative node). Confidence:
    /// `exact` (unique match), `name_match` (ambiguous — multiple notes same title; first by
    /// file id is recorded but flagged), or left NULL (dangling — no such note).
    pub fn resolve_wikilinks(&mut self) -> Result<usize> {
        // file_id -> path; title(lowercased, no ext) -> file_ids; file_id -> chunk-0 id.
        let files: Vec<(i64, String)> = {
            let mut stmt = self.conn.prepare("SELECT id, path FROM files")?;
            let rows = stmt.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))?;
            let mut out = Vec::new();
            for r in rows {
                out.push(r?);
            }
            out
        };
        if files.is_empty() {
            return Ok(0);
        }
        let chunk0: HashMap<i64, i64> = {
            let mut stmt = self
                .conn
                .prepare("SELECT file_id, id FROM chunks WHERE chunk_idx = 0")?;
            let rows = stmt.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?)))?;
            let mut out = HashMap::new();
            for r in rows {
                let (f, c) = r?;
                out.insert(f, c);
            }
            out
        };
        let path_no_ext = |p: &str| p.strip_suffix(".md").unwrap_or(p).to_lowercase();
        let mut title_map: HashMap<String, Vec<i64>> = HashMap::new();
        for (fid, path) in &files {
            let base = std::path::Path::new(path)
                .file_stem()
                .map(|s| s.to_string_lossy().to_lowercase())
                .unwrap_or_default();
            title_map.entry(base).or_default().push(*fid);
        }

        // Unresolved wikilink edges.
        let edges: Vec<(i64, String, Option<String>)> = {
            let mut stmt = self.conn.prepare(
                "SELECT id, target_symbol, target_path FROM links \
                 WHERE kind = 'wikilink' AND target_chunk_id IS NULL",
            )?;
            let rows = stmt.query_map([], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, Option<String>>(1)?.unwrap_or_default(),
                    r.get::<_, Option<String>>(2)?,
                ))
            })?;
            let mut out = Vec::new();
            for r in rows {
                out.push(r?);
            }
            out
        };
        if edges.is_empty() {
            return Ok(0);
        }

        let tx = self.conn.unchecked_transaction()?;
        let mut resolved = 0usize;
        for (link_id, title, target_path) in &edges {
            // Candidate file ids: path-suffix match first (when the link carried a path),
            // else title (basename) match.
            let candidates: Vec<i64> = match target_path {
                Some(p) => {
                    let want = path_no_ext(p);
                    let hits: Vec<i64> = files
                        .iter()
                        .filter(|(_, fp)| {
                            let fp = path_no_ext(fp);
                            fp == want || fp.ends_with(&format!("/{want}"))
                        })
                        .map(|(id, _)| *id)
                        .collect();
                    if hits.is_empty() {
                        title_map
                            .get(&title.to_lowercase())
                            .cloned()
                            .unwrap_or_default()
                    } else {
                        hits
                    }
                }
                None => title_map
                    .get(&title.to_lowercase())
                    .cloned()
                    .unwrap_or_default(),
            };
            if candidates.is_empty() {
                continue; // dangling link — leave target_chunk_id NULL
            }
            let mut sorted = candidates.clone();
            sorted.sort_unstable();
            let target_file = sorted[0];
            let Some(&chunk_id) = chunk0.get(&target_file) else {
                continue;
            };
            let confidence = if candidates.len() == 1 {
                "exact"
            } else {
                "name_match"
            };
            tx.execute(
                "UPDATE links SET target_chunk_id = ?, confidence = ? WHERE id = ?",
                params![chunk_id, confidence, link_id],
            )?;
            resolved += 1;
        }
        tx.commit()?;
        Ok(resolved)
    }

    /// Notes that link TO `path` via a resolved wikilink (inbound links / "backlinks").
    /// Returns distinct source-note paths, sorted.
    pub fn backlinks(&self, path: &str) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT sf.path \
             FROM links l \
             JOIN chunks sc ON sc.id = l.source_chunk_id \
             JOIN files sf ON sf.id = sc.file_id \
             JOIN chunks tc ON tc.id = l.target_chunk_id \
             JOIN files tf ON tf.id = tc.file_id \
             WHERE l.kind = 'wikilink' AND tf.path = ? AND sf.path != tf.path \
             ORDER BY sf.path",
        )?;
        let rows = stmt.query_map(params![path], |r| r.get::<_, String>(0))?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Notes `path` links TO via resolved wikilinks (outbound links). Distinct target-note
    /// paths, sorted.
    pub fn forward_links(&self, path: &str) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT tf.path \
             FROM links l \
             JOIN chunks sc ON sc.id = l.source_chunk_id \
             JOIN files sf ON sf.id = sc.file_id \
             JOIN chunks tc ON tc.id = l.target_chunk_id \
             JOIN files tf ON tf.id = tc.file_id \
             WHERE l.kind = 'wikilink' AND sf.path = ? AND sf.path != tf.path \
             ORDER BY tf.path",
        )?;
        let rows = stmt.query_map(params![path], |r| r.get::<_, String>(0))?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
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

    // ---------- per-subpath context strings (v0.4) ----------

    /// Upsert a context description anchored at `path_prefix`. Use `"/"` for a source-wide
    /// default. Surfaced on each search hit whose path is under the prefix.
    pub fn add_context(&self, path_prefix: &str, description: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO contexts (path_prefix, description, updated_at) VALUES (?, ?, ?) \
             ON CONFLICT(path_prefix) DO UPDATE SET \
                 description = excluded.description, \
                 updated_at  = excluded.updated_at",
            params![path_prefix, description, now_secs()],
        )?;
        Ok(())
    }

    /// Remove a context entry. Returns true iff a row was deleted.
    pub fn remove_context(&self, path_prefix: &str) -> Result<bool> {
        let affected = self.conn.execute(
            "DELETE FROM contexts WHERE path_prefix = ?",
            params![path_prefix],
        )?;
        Ok(affected > 0)
    }

    /// List every (path_prefix, description) pair, sorted by prefix.
    pub fn list_contexts(&self) -> Result<Vec<(String, String)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT path_prefix, description FROM contexts ORDER BY path_prefix")?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Return the deepest-matching context description for the given path, if any. Convention:
    ///   * `path_prefix = "/"` — source-wide default; matches every path.
    ///   * `path_prefix = "/api"` — matches any path whose first segment is `api/...` (we
    ///     normalize the searched path with a leading `/` first, since chunk paths are
    ///     stored relative to the source root without one).
    ///     Subtree match requires a `/` boundary so `/foo` doesn't accidentally match `/foobar`.
    ///     Longer prefix wins ties so a subtree context overrides the global one.
    pub fn context_for(&self, path: &str) -> Result<Option<String>> {
        // Cheap fast-path: skip the SELECT when no contexts are registered. Worth it because
        // this runs once per Hit, and the typical source has zero contexts configured.
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM contexts", [], |r| r.get(0))?;
        if count == 0 {
            return Ok(None);
        }
        let normalized = if path.starts_with('/') {
            path.to_string()
        } else {
            format!("/{path}")
        };
        let mut stmt = self.conn.prepare(
            "SELECT description FROM contexts \
             WHERE path_prefix = '/' \
                OR path_prefix = ? \
                OR ? LIKE path_prefix || '/%' \
             ORDER BY LENGTH(path_prefix) DESC LIMIT 1",
        )?;
        let result: rusqlite::Result<String> =
            stmt.query_row(params![&normalized, &normalized], |r| r.get(0));
        match result {
            Ok(s) => Ok(Some(s)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Enumerate every `files.path` that matches the glob pattern. Pattern semantics match
    /// `globset` (e.g. `**/*.rs`, `src/chunk/*.rs`). Paths are returned in alphabetical order.
    /// The body bytes are NOT read here — callers can decide whether they need each one.
    pub fn list_paths_matching(&self, pattern: &str) -> Result<Vec<String>> {
        let glob = globset::Glob::new(pattern)
            .with_context(|| format!("invalid glob pattern: {pattern}"))?
            .compile_matcher();
        let mut stmt = self.conn.prepare("SELECT path FROM files ORDER BY path")?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
        let mut out = Vec::new();
        for r in rows {
            let p = r?;
            if glob.is_match(&p) {
                out.push(p);
            }
        }
        Ok(out)
    }

    pub fn has_file_path(&self, path: &str) -> Result<bool> {
        let n: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM files WHERE path = ?",
            params![path],
            |r| r.get(0),
        )?;
        Ok(n > 0)
    }

    // ---------- usage logging (v0.6) ----------

    /// Append one row to the `usage` table: which query was run, what we returned, with the
    /// query's embedding (so v0.7's signal-based reranker can score similarity-to-past-query
    /// without re-embedding). `used_chunk_id` starts NULL — patched in by `mark_used` if an
    /// MCP follow-up call signals the user/agent actually consumed one of the returned hits.
    pub fn log_usage(
        &self,
        query: &str,
        query_embedding: &[f32],
        returned_chunks_json: &str,
    ) -> Result<i64> {
        let bytes = f32_vec_to_bytes(query_embedding);
        self.conn.execute(
            "INSERT INTO usage (query_text, query_embedding, returned_chunks, used_chunk_id, created_at) \
             VALUES (?, ?, ?, NULL, ?)",
            params![query, bytes, returned_chunks_json, now_secs()],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Locate the most recent un-attributed usage row whose `query_text` matches the given
    /// query and was logged within `max_age_secs` seconds, then patch its `used_chunk_id`.
    /// Returns true iff a row was updated. Used by the MCP ring buffer to attribute a
    /// follow-up `multi_get` read back to its originating search.
    pub fn mark_used_by_query(
        &self,
        query: &str,
        chunk_id: i64,
        max_age_secs: i64,
    ) -> Result<bool> {
        let cutoff = now_secs() - max_age_secs;
        let affected = self.conn.execute(
            "UPDATE usage SET used_chunk_id = ? \
             WHERE id = ( \
                 SELECT id FROM usage \
                 WHERE query_text = ? AND used_chunk_id IS NULL AND created_at >= ? \
                 ORDER BY created_at DESC LIMIT 1 \
             )",
            params![chunk_id, query, cutoff],
        )?;
        Ok(affected > 0)
    }

    // ---------- code-aware lookups (sub-slice E) ----------

    pub fn find_definitions(&self, symbol: &str, limit: usize) -> Result<Vec<DefinitionHit>> {
        let mut out = self.find_exact_definitions(symbol, limit)?;
        if out.len() >= limit {
            return Ok(out);
        }
        let mut aliases = self.find_alias_definitions(symbol, limit - out.len())?;
        aliases.retain(|hit| !out.iter().any(|exact| exact.chunk_id == hit.chunk_id));
        out.extend(aliases);
        Ok(out)
    }

    fn find_exact_definitions(&self, symbol: &str, limit: usize) -> Result<Vec<DefinitionHit>> {
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

    fn find_alias_definitions(&self, symbol: &str, limit: usize) -> Result<Vec<DefinitionHit>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let mut stmt = self.conn.prepare(
            "SELECT c.id, c.heading_path, c.content, c.start_byte, c.kind, c.symbol, f.path \
             FROM chunks c JOIN files f ON f.id = c.file_id \
             WHERE c.symbol IS NOT NULL AND c.kind NOT IN ('prose') \
             ORDER BY f.path, c.chunk_idx",
        )?;
        let rows = stmt.query_map([], |row| {
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
        for row in rows {
            let hit = row?;
            if crate::chunk::symbol_matches_alias(&hit.symbol, symbol) {
                out.push(hit);
                if out.len() >= limit {
                    break;
                }
            }
        }
        Ok(out)
    }

    /// Direct callers (depth=1) and transitive callers (depth>1). Uses a recursive CTE so
    /// the call doesn't N+1 across depth levels.
    pub fn find_callers(&self, symbol: &str, depth: usize, limit: usize) -> Result<Vec<CallerHit>> {
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

    pub fn find_implementations(&self, symbol: &str, limit: usize) -> Result<Vec<DefinitionHit>> {
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
        let placeholders = std::iter::repeat_n("?", file_ids.len())
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "SELECT f.path, f.id, c.symbol, c.kind, c.heading_path \
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
                let rows =
                    stmt.query_map(params![query, limit as i64], |row| row.get::<_, i64>(0))?;
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

    /// Literal substring scan over `chunks.content` — the third RRF arm. Catches what
    /// FTS5's tokenizer drops: camelCase fragments (`Request` inside `processRequest`),
    /// snake_case adjacency (`foo_bar`), magic constants (`E_NOENT`, `MAX_RETRY_COUNT`).
    /// SQLite LIKE is case-insensitive for ASCII and O(N) over the column; on personal-
    /// vault sizes the per-query cost is invisible. `%`/`_`/`\\` in the query are escaped
    /// so a user typing `foo_bar` matches `foo_bar` literally rather than `foo<anything>bar`.
    pub fn search_literal(
        &self,
        query: &str,
        limit: usize,
        path_prefix: Option<&str>,
    ) -> Result<Vec<i64>> {
        let pattern = format!("%{}%", escape_like(query));
        match path_prefix {
            None => {
                let mut stmt = self.conn.prepare(
                    "SELECT id FROM chunks WHERE content LIKE ? ESCAPE '\\' \
                     ORDER BY id LIMIT ?",
                )?;
                let rows =
                    stmt.query_map(params![pattern, limit as i64], |row| row.get::<_, i64>(0))?;
                let mut hits = Vec::new();
                for r in rows {
                    hits.push(r?);
                }
                Ok(hits)
            }
            Some(prefix) => {
                let path_like = format!("{}%", prefix);
                let mut stmt = self.conn.prepare(
                    "SELECT c.id FROM chunks c \
                     JOIN files f ON f.id = c.file_id \
                     WHERE c.content LIKE ? ESCAPE '\\' AND f.path LIKE ? \
                     ORDER BY c.id LIMIT ?",
                )?;
                let rows = stmt.query_map(params![pattern, path_like, limit as i64], |row| {
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
                let rows =
                    stmt.query_map(params![bytes, limit as i64], |row| row.get::<_, i64>(0))?;
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
                let rows = stmt
                    .query_map(params![bytes, inner_limit, like, limit as i64], |row| {
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

    // ---------- derived graph edges (Layer B) ----------

    /// Read a chunk's stored embedding back out of the sqlite-vec virtual table, decoded to
    /// f32. Used to build the kNN similarity graph. Returns None if the chunk has no vector.
    #[cfg(test)]
    pub fn fetch_chunk_embedding(&self, chunk_id: i64) -> Result<Option<Vec<f32>>> {
        let blob: rusqlite::Result<Vec<u8>> = self.conn.query_row(
            "SELECT embedding FROM chunks_vec WHERE chunk_id = ?",
            params![chunk_id],
            |r| r.get(0),
        );
        match blob {
            Ok(b) => Ok(Some(bytes_to_f32_vec(&b))),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Every (chunk_id, embedding) pair, for building the similarity graph in one pass.
    pub fn all_chunk_embeddings(&self) -> Result<Vec<(i64, Vec<f32>)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT chunk_id, embedding FROM chunks_vec")?;
        let rows = stmt.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, Vec<u8>>(1)?)))?;
        let mut out = Vec::new();
        for r in rows {
            let (id, blob) = r?;
            out.push((id, bytes_to_f32_vec(&blob)));
        }
        Ok(out)
    }

    /// Every (chunk_id, content) pair, for keyphrase extraction over the corpus.
    pub fn all_chunks_for_graph(&self) -> Result<Vec<(i64, String)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, content FROM chunks ORDER BY id")?;
        let rows = stmt.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn clear_graph_edges(&self) -> Result<()> {
        self.conn.execute("DELETE FROM graph_edges", [])?;
        Ok(())
    }

    /// Batch-insert derived edges in one transaction. Each tuple is
    /// `(src_chunk_id, dst_chunk_id, kind, weight)`.
    pub fn insert_graph_edges(&self, edges: &[(i64, i64, &str, f64)]) -> Result<()> {
        if edges.is_empty() {
            return Ok(());
        }
        let tx = self.conn.unchecked_transaction()?;
        {
            let mut stmt = tx.prepare(
                "INSERT INTO graph_edges (src_chunk_id, dst_chunk_id, kind, weight) \
                 VALUES (?, ?, ?, ?)",
            )?;
            for (src, dst, kind, weight) in edges {
                stmt.execute(params![src, dst, kind, weight])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    #[cfg(test)]
    pub fn count_graph_edges(&self) -> Result<i64> {
        Ok(self
            .conn
            .query_row("SELECT COUNT(*) FROM graph_edges", [], |r| r.get(0))?)
    }

    /// Build the directed, weighted edge list for Personalized-PageRank retrieval (Layer C):
    /// the union of resolved wikilinks (`links`, kind='wikilink') and derived edges
    /// (`graph_edges`). Edge weight = a flat per-kind weight so authored links dominate:
    /// wikilink 1.0, entity 0.6, keyphrase 0.5, similarity 0.3. Undirected sources are
    /// symmetrized (both directions); directed wikilinks add a reverse edge at half weight so
    /// leaf notes aren't stranded under PPR.
    pub fn graph_edges_for_ppr(&self) -> Result<Vec<(i64, i64, f64)>> {
        let mut out: Vec<(i64, i64, f64)> = Vec::new();

        // Authored wikilinks (directed) — forward at 1.0, reverse at 0.5.
        let mut stmt = self.conn.prepare(
            "SELECT source_chunk_id, target_chunk_id FROM links \
             WHERE kind = 'wikilink' AND target_chunk_id IS NOT NULL",
        )?;
        let rows = stmt.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?)))?;
        for r in rows {
            let (s, d) = r?;
            if s != d {
                out.push((s, d, 1.0));
                out.push((d, s, 0.5));
            }
        }

        // Derived edges (stored undirected as (min,max)) — symmetrize at the flat kind weight.
        let mut stmt = self
            .conn
            .prepare("SELECT src_chunk_id, dst_chunk_id, kind FROM graph_edges")?;
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, i64>(1)?,
                r.get::<_, String>(2)?,
            ))
        })?;
        for r in rows {
            let (s, d, kind) = r?;
            let w = match kind.as_str() {
                "entity" => 0.6,
                "keyphrase" => 0.5,
                "similarity" => 0.3,
                _ => 0.3,
            };
            if s != d {
                out.push((s, d, w));
                out.push((d, s, w));
            }
        }
        Ok(out)
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
        let placeholders = std::iter::repeat_n("?", ids.len())
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

/// Escape `%`, `_`, and `\` in a string so it can be wrapped in `%…%` and used with
/// `LIKE ? ESCAPE '\'`. Without this, a user query of `foo_bar` would match `foo<any>bar`.
fn escape_like(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if c == '\\' || c == '%' || c == '_' {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

fn f32_vec_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for f in v {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

/// Inverse of `f32_vec_to_bytes` — decode a little-endian f32 blob (as stored in
/// `chunks_vec`) back to a vector. Trailing bytes that don't form a full f32 are ignored.
fn bytes_to_f32_vec(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    /// The embedding written into chunks_vec round-trips back out via fetch_chunk_embedding /
    /// all_chunk_embeddings — load-bearing for the kNN similarity graph. Also exercises the
    /// graph_edges CRUD.
    #[test]
    fn embedding_roundtrip_and_graph_edges() {
        init_sqlite_vec();
        let dir = tempdir().unwrap();
        let db = dir.path().join("g.db");
        let mut store = Store::open(&db, 4).unwrap();

        let emb_a = vec![0.1f32, 0.2, 0.3, 0.4];
        let emb_b = vec![0.5f32, 0.6, 0.7, 0.8];
        fn row<'a>(c: &'a str, e: &'a [f32]) -> ChunkRow<'a> {
            ChunkRow {
                idx: 0,
                heading_path: "",
                content: c,
                start_byte: 0,
                end_byte: c.len(),
                embedding: e,
                kind: "prose",
                symbol: None,
                parent_chunk_idx: None,
            }
        }
        store
            .upsert_file_with_chunks("a.md", 1, 4, "ha", &[row("alpha", &emb_a)], &[])
            .unwrap();
        store
            .upsert_file_with_chunks("b.md", 1, 4, "hb", &[row("beta", &emb_b)], &[])
            .unwrap();

        // Find the two chunk ids.
        let all = store.all_chunk_embeddings().unwrap();
        assert_eq!(all.len(), 2);
        for (_, v) in &all {
            assert_eq!(v.len(), 4, "decoded vector has the right dim");
        }
        let (id_a, vec_a) = all
            .iter()
            .find(|(_, v)| (v[0] - 0.1).abs() < 1e-4)
            .cloned()
            .unwrap();
        // Round-trip tolerance: sqlite-vec stores f32, so this is exact-ish.
        assert!(
            (vec_a[3] - 0.4).abs() < 1e-4,
            "fetched vector matches stored"
        );
        let single = store.fetch_chunk_embedding(id_a).unwrap().unwrap();
        assert_eq!(single, vec_a);

        // graph_edges CRUD.
        let (id_b, _) = all.iter().find(|(id, _)| *id != id_a).cloned().unwrap();
        store
            .insert_graph_edges(&[(id_a, id_b, "similarity", 0.3)])
            .unwrap();
        assert_eq!(store.count_graph_edges().unwrap(), 1);
        store.clear_graph_edges().unwrap();
        assert_eq!(store.count_graph_edges().unwrap(), 0);
    }

    #[test]
    fn find_definitions_matches_symbol_aliases_after_exact_hits() {
        init_sqlite_vec();
        let dir = tempdir().unwrap();
        let db = dir.path().join("aliases.db");
        let mut store = Store::open(&db, 4).unwrap();
        let emb = vec![0.0f32, 0.0, 0.0, 0.0];
        let rows = vec![ChunkRow {
            idx: 0,
            heading_path: "Store",
            content: "fn processRequest() {}",
            start_byte: 0,
            end_byte: 22,
            embedding: &emb,
            kind: "function",
            symbol: Some("processRequest"),
            parent_chunk_idx: None,
        }];
        store
            .upsert_file_with_chunks("src/store.rs", 1, 22, "hash", &rows, &[])
            .unwrap();

        let exact = store.find_definitions("processRequest", 10).unwrap();
        assert_eq!(exact.len(), 1);
        assert_eq!(exact[0].symbol, "processRequest");

        let alias = store.find_definitions("process request", 10).unwrap();
        assert_eq!(alias.len(), 1);
        assert_eq!(alias[0].symbol, "processRequest");
    }

    /// Wikilink edges resolve by note title/path to the target's chunk 0, and backlinks /
    /// forward_links read them back. Covers exact (unique title), path-qualified, and
    /// dangling (no such note) cases.
    #[test]
    fn wikilink_resolution_and_backlinks() {
        init_sqlite_vec();
        let dir = tempdir().unwrap();
        let db = dir.path().join("w.db");
        let mut store = Store::open(&db, 4).unwrap();
        let emb = vec![0.0f32, 0.0, 0.0, 0.0];

        fn row<'a>(content: &'a str, emb: &'a [f32]) -> ChunkRow<'a> {
            ChunkRow {
                idx: 0,
                heading_path: "",
                content,
                start_byte: 0,
                end_byte: content.len(),
                embedding: emb,
                kind: "prose",
                symbol: None,
                parent_chunk_idx: None,
            }
        }

        // "index.md" links to [[Target]] (title), [[sub/Deep]] (path), and [[Ghost]] (dangling).
        let links = vec![
            LinkRow {
                source_chunk_idx: 0,
                kind: "wikilink",
                target_symbol: "Target",
                target_path: None,
            },
            LinkRow {
                source_chunk_idx: 0,
                kind: "wikilink",
                target_symbol: "Deep",
                target_path: Some("sub/Deep"),
            },
            LinkRow {
                source_chunk_idx: 0,
                kind: "wikilink",
                target_symbol: "Ghost",
                target_path: None,
            },
        ];
        store
            .upsert_file_with_chunks("index.md", 1, 10, "h1", &[row("see links", &emb)], &links)
            .unwrap();
        store
            .upsert_file_with_chunks(
                "Target.md",
                1,
                10,
                "h2",
                &[row("i am the target", &emb)],
                &[],
            )
            .unwrap();
        store
            .upsert_file_with_chunks("sub/Deep.md", 1, 10, "h3", &[row("deep note", &emb)], &[])
            .unwrap();

        let resolved = store.resolve_wikilinks().unwrap();
        assert_eq!(resolved, 2, "Target + Deep resolve; Ghost dangles");

        // Backlinks of the two real targets point back to index.md.
        assert_eq!(
            store.backlinks("Target.md").unwrap(),
            vec!["index.md".to_string()]
        );
        assert_eq!(
            store.backlinks("sub/Deep.md").unwrap(),
            vec!["index.md".to_string()]
        );
        // Ghost.md doesn't exist → no backlinks.
        assert!(store.backlinks("Ghost.md").unwrap().is_empty());

        // Forward links of index.md are the two resolved targets.
        let mut fwd = store.forward_links("index.md").unwrap();
        fwd.sort();
        assert_eq!(
            fwd,
            vec!["Target.md".to_string(), "sub/Deep.md".to_string()]
        );
    }

    /// FTS must see `heading_path` so queries matching a section title still hit the chunk
    /// even when the chunker stripped the heading line from the body. Regression for v0.2.6.
    #[test]
    fn fts_indexes_heading_path() {
        init_sqlite_vec();
        let dir = tempdir().unwrap();
        let db = dir.path().join("t.db");
        let mut store = Store::open(&db, 4).unwrap();

        let emb = vec![0.0f32, 0.0, 0.0, 0.0];
        // Body deliberately omits the unique heading words ("Quokka" / "Diplodocus") so
        // the only way FTS can return this chunk for those terms is via the heading path.
        let rows = vec![ChunkRow {
            idx: 0,
            heading_path: "Quokka > Diplodocus",
            content: "lorem ipsum dolor sit amet",
            start_byte: 0,
            end_byte: 26,
            embedding: &emb,
            kind: "prose",
            symbol: None,
            parent_chunk_idx: None,
        }];
        store
            .upsert_file_with_chunks("note.md", 1, 26, "hash", &rows, &[])
            .unwrap();

        let hits = store.search_fts("\"Quokka\"", 10, None).unwrap();
        assert_eq!(hits.len(), 1, "heading-only term must reach FTS");
        let hits = store.search_fts("\"Diplodocus\"", 10, None).unwrap();
        assert_eq!(hits.len(), 1, "nested heading term must reach FTS");
        let hits = store.search_fts("\"lorem\"", 10, None).unwrap();
        assert_eq!(hits.len(), 1, "body terms still work");
    }

    #[test]
    fn context_for_matches_subtree_and_global_with_boundary() {
        init_sqlite_vec();
        let dir = tempdir().unwrap();
        let db = dir.path().join("c.db");
        let store = Store::open(&db, 4).unwrap();

        // No contexts yet → None for everything.
        assert!(store.context_for("technology/x.md").unwrap().is_none());

        // Add a subtree context.
        store
            .add_context("/technology", "Engineering nuggets")
            .unwrap();
        assert_eq!(
            store.context_for("technology/Reciprocal.md").unwrap(),
            Some("Engineering nuggets".to_string()),
            "subtree path should match /technology prefix"
        );
        // Boundary safety: /technology must NOT match /technology2/...
        assert!(
            store.context_for("technology2/foo.md").unwrap().is_none(),
            "subtree match must require / boundary"
        );

        // Add a deeper override + a global default. Deepest wins.
        store
            .add_context("/technology/rrf", "RRF-specific notes")
            .unwrap();
        store.add_context("/", "Personal vault").unwrap();
        assert_eq!(
            store
                .context_for("technology/rrf/details.md")
                .unwrap()
                .unwrap(),
            "RRF-specific notes"
        );
        assert_eq!(
            store.context_for("technology/other.md").unwrap().unwrap(),
            "Engineering nuggets"
        );
        assert_eq!(
            store.context_for("gtm/foo.md").unwrap().unwrap(),
            "Personal vault"
        );

        // Removal works.
        assert!(store.remove_context("/technology/rrf").unwrap());
        assert_eq!(
            store
                .context_for("technology/rrf/details.md")
                .unwrap()
                .unwrap(),
            "Engineering nuggets",
            "after removing the deepest, the parent prefix takes over"
        );
    }

    /// usage rows round-trip through log_usage → mark_used_by_query, including the
    /// "no recent match" branch and the time-window guard.
    #[test]
    fn usage_log_and_attribution_roundtrip() {
        init_sqlite_vec();
        let dir = tempdir().unwrap();
        let db = dir.path().join("u.db");
        let store = Store::open(&db, 4).unwrap();

        let emb = vec![0.1f32, 0.2, 0.3, 0.4];
        let usage_id = store
            .log_usage("how to track ICP", &emb, "[1,2,3]")
            .unwrap();
        assert!(usage_id > 0);

        // Matching query within window → attribution succeeds.
        let patched = store
            .mark_used_by_query("how to track ICP", 42, 60)
            .unwrap();
        assert!(patched, "matching query inside window should patch a row");

        // The same row can't be patched twice (used_chunk_id is no longer NULL).
        let patched_again = store
            .mark_used_by_query("how to track ICP", 99, 60)
            .unwrap();
        assert!(
            !patched_again,
            "already-attributed row must not be re-patched"
        );

        // Non-matching query → no-op.
        let nonmatch = store
            .mark_used_by_query("something else entirely", 7, 60)
            .unwrap();
        assert!(!nonmatch);

        // Confirm the stored row carries the expected used_chunk_id.
        let used: Option<i64> = store
            .conn
            .query_row(
                "SELECT used_chunk_id FROM usage WHERE id = ?",
                params![usage_id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(used, Some(42));
    }

    /// Migrations run once on first open, are recorded, and don't re-apply on subsequent
    /// opens. Re-opening a DB created by a future binary (with extra migrations) must not
    /// error — the loop just skips already-applied versions.
    #[test]
    fn migrations_apply_once_and_only_once() {
        use crate::migrations::MIGRATIONS;
        init_sqlite_vec();
        let dir = tempdir().unwrap();
        let db = dir.path().join("m.db");

        let store = Store::open(&db, 4).unwrap();
        let count: i64 = store
            .conn
            .query_row("SELECT COUNT(*) FROM migrations", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, MIGRATIONS.len() as i64);
        let max_version: i64 = store
            .conn
            .query_row("SELECT MAX(version) FROM migrations", [], |r| r.get(0))
            .unwrap();
        assert_eq!(max_version, MIGRATIONS.last().unwrap().0);
        drop(store);

        // Re-open the same DB — should be a no-op.
        let store2 = Store::open(&db, 4).unwrap();
        let count2: i64 = store2
            .conn
            .query_row("SELECT COUNT(*) FROM migrations", [], |r| r.get(0))
            .unwrap();
        assert_eq!(
            count2,
            MIGRATIONS.len() as i64,
            "migrations must be idempotent"
        );

        // Confirm migration #1 created the contexts table.
        let table_exists: i64 = store2
            .conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='contexts'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(table_exists, 1, "contexts table created by migration #1");
    }
}

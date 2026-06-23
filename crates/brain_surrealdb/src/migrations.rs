use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use jiff::Timestamp;
use surrealdb::{Surreal, engine::any::Any};
use surrealdb_types::Value;

pub struct MigrationRunner {
    db: Surreal<Any>,
    migrations_dir: PathBuf,
}

impl MigrationRunner {
    pub fn new(db: Surreal<Any>, migrations_dir: impl Into<PathBuf>) -> Self {
        Self {
            db,
            migrations_dir: migrations_dir.into(),
        }
    }

    pub async fn run(&self) -> Result<()> {
        self.ensure_migrations_table().await?;
        let applied = self.applied_filenames().await?;
        let pending = self.pending_migrations(&applied)?;

        for path in pending {
            let filename = path
                .file_name()
                .and_then(|n| n.to_str())
                .map(|s| s.to_string())
                .context("migration filename is not valid UTF-8")?;

            let sql = std::fs::read_to_string(&path)
                .with_context(|| format!("failed to read migration file: {}", filename))?;

            self.db
                .query(&sql)
                .await
                .with_context(|| format!("failed to execute migration: {}", filename))?;

            self.record_applied(&filename).await?;
        }

        Ok(())
    }

    async fn ensure_migrations_table(&self) -> Result<()> {
        self.db
            .query(
                r#"
                DEFINE TABLE IF NOT EXISTS _migrations SCHEMAFULL;
                DEFINE FIELD IF NOT EXISTS filename ON _migrations TYPE string;
                DEFINE FIELD IF NOT EXISTS applied_at ON _migrations TYPE string;
                DEFINE INDEX IF NOT EXISTS idx_filename ON _migrations COLUMNS filename UNIQUE;
                "#,
            )
            .await?;
        Ok(())
    }

    async fn applied_filenames(&self) -> Result<std::collections::HashSet<String>> {
        let mut response = self.db.query("SELECT filename FROM _migrations").await?;
        let rows: Vec<Value> = response.take(0)?;
        let filenames = rows
            .into_iter()
            .filter_map(|row| {
                row.into_object()
                    .ok()
                    .and_then(|obj| obj.get("filename").cloned())
                    .and_then(|v| v.into_string().ok())
            })
            .collect();
        Ok(filenames)
    }

    fn pending_migrations(
        &self,
        applied: &std::collections::HashSet<String>,
    ) -> Result<Vec<PathBuf>> {
        let mut pending: Vec<PathBuf> = std::fs::read_dir(&self.migrations_dir)
            .with_context(|| {
                format!(
                    "failed to read migrations directory: {}",
                    self.migrations_dir.display()
                )
            })?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| path.extension() == Some(OsStr::new("surql")) && path.is_file())
            .filter(|path| {
                path.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| !applied.contains(n))
                    .unwrap_or(false)
            })
            .collect();

        pending.sort();
        Ok(pending)
    }

    async fn record_applied(&self, filename: &str) -> Result<()> {
        let now = Timestamp::now().to_string();
        self.db
            .query("CREATE _migrations SET filename = $filename, applied_at = $applied_at")
            .bind(("filename", filename))
            .bind(("applied_at", now))
            .await?;
        Ok(())
    }
}

pub fn default_migrations_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("migrations")
}

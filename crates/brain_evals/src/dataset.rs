//! Loading of versioned golden datasets.
//!
//! A dataset is a JSON Lines file named `<action>.<version>.jsonl` (for example
//! `detect_topic_shift.v1.jsonl`). The version is part of the filename so a
//! revision to the cases is a visible, reviewable diff rather than a mutation
//! of an existing file. Blank lines and lines beginning with `#` are ignored,
//! so a file can carry a header comment.

use std::path::Path;

use anyhow::{Context, Result};
use serde::de::DeserializeOwned;

/// A versioned collection of evaluation cases for a single action.
///
/// Generic over the case type so each action defines its own schema while
/// sharing one loader. The cases hold inputs and *expectations*; the results of
/// running them live in a [`crate::Report`], never in the dataset, because
/// outputs are stochastic and the dataset must stay stable across runs.
#[derive(Debug, Clone)]
pub struct Dataset<C> {
    /// The action these cases exercise, e.g. `detect_topic_shift`.
    pub action: String,
    /// The dataset version, e.g. `v1`.
    pub version: String,
    pub cases: Vec<C>,
}

impl<C: DeserializeOwned> Dataset<C> {
    /// Parses cases from JSON Lines content, one case per non-empty,
    /// non-comment line.
    pub fn from_jsonl(
        action: impl Into<String>,
        version: impl Into<String>,
        contents: &str,
    ) -> Result<Self> {
        let mut cases = Vec::new();
        for (index, raw) in contents.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let case = serde_json::from_str(line)
                .with_context(|| format!("failed to parse case on line {}", index + 1))?;
            cases.push(case);
        }
        Ok(Self {
            action: action.into(),
            version: version.into(),
            cases,
        })
    }

    /// Loads a dataset from a `<action>.<version>.jsonl` file, inferring the
    /// action and version from the filename.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let (action, version) = parse_stem(path).with_context(|| {
            format!(
                "dataset filename must be '<action>.<version>.jsonl': {}",
                path.display()
            )
        })?;
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read dataset {}", path.display()))?;
        Self::from_jsonl(action, version, &contents)
    }

    pub fn len(&self) -> usize {
        self.cases.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cases.is_empty()
    }
}

/// Splits `<action>.<version>.jsonl` into its action and version parts.
fn parse_stem(path: &Path) -> Option<(String, String)> {
    let name = path.file_name()?.to_str()?;
    let stem = name.strip_suffix(".jsonl")?;
    let (action, version) = stem.rsplit_once('.')?;
    if action.is_empty() || version.is_empty() {
        return None;
    }
    Some((action.to_string(), version.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Deserialize)]
    struct Probe {
        id: String,
    }

    #[test]
    fn from_jsonl_skips_blank_and_comment_lines() {
        let contents = "# header comment\n\n{\"id\":\"a\"}\n{\"id\":\"b\"}\n";
        let dataset: Dataset<Probe> = Dataset::from_jsonl("probe", "v1", contents).unwrap();

        assert_eq!(dataset.action, "probe");
        assert_eq!(dataset.version, "v1");
        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.cases[0].id, "a");
        assert_eq!(dataset.cases[1].id, "b");
    }

    #[test]
    fn from_jsonl_reports_the_offending_line() {
        let contents = "{\"id\":\"a\"}\nnot json\n";
        let error = Dataset::<Probe>::from_jsonl("probe", "v1", contents).unwrap_err();
        assert!(error.to_string().contains("line 2"));
    }

    #[test]
    fn parse_stem_extracts_action_and_version() {
        let parsed = parse_stem(Path::new("datasets/detect_topic_shift.v1.jsonl"));
        assert_eq!(
            parsed,
            Some(("detect_topic_shift".to_string(), "v1".to_string()))
        );
    }

    #[test]
    fn parse_stem_rejects_unversioned_names() {
        assert!(parse_stem(Path::new("detect_topic_shift.jsonl")).is_none());
    }
}

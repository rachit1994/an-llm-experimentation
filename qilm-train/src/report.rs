//! report — regenerate `results/RESULTS.md` / `results/RESULTS.json` from
//! `runs/` + `gates.toml`, checking provenance and refusing to emit
//! fabricated/stale numbers (T0.9, VERIFICATION.md §4.2).
//!
//! This module is the testable core behind `src/bin/report.rs`; the binary
//! is a thin CLI wrapper so `generate`/`check` can be exercised directly by
//! `qilm-train/tests/report_harness.rs` without spawning a subprocess.

use crate::gate::{evaluate, load_gates, GateOutcome};
use crate::provenance::{code_fingerprint, current_git_sha, RunRecord};
use serde::Serialize;
use std::fs;
use std::path::Path;

pub struct ReportConfig<'a> {
    pub runs_dir: &'a Path,
    pub gates_toml: &'a Path,
    pub results_dir: &'a Path,
    pub workspace_root: &'a Path,
}

#[derive(Debug)]
pub enum ReportError {
    Io(std::io::Error),
    Provenance(String),
    Gate(String),
}
impl std::fmt::Display for ReportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReportError::Io(e) => write!(f, "io error: {e}"),
            ReportError::Provenance(m) => write!(f, "provenance check failed: {m}"),
            ReportError::Gate(m) => write!(f, "gate error: {m}"),
        }
    }
}
impl std::error::Error for ReportError {}
impl From<std::io::Error> for ReportError {
    fn from(e: std::io::Error) -> Self {
        ReportError::Io(e)
    }
}

#[derive(Debug, Serialize)]
struct RunEntry {
    run_id: String,
    git_sha: String,
    dataset_sha256: String,
    seed: u64,
    backend: String,
    metrics: serde_json::Value,
    gates: Vec<GateEntry>,
}

#[derive(Debug, Serialize)]
struct GateEntry {
    name: String,
    outcome: String, // "PASS" | "FAIL" | "N/A"
    measured: Option<f64>,
    threshold: Option<f64>,
}

#[derive(Debug, Serialize)]
struct ResultsJson {
    runs: Vec<RunEntry>,
}

const GATE_NAMES: &[&str] = &[
    "gradcheck",
    "g0",
    "collapse",
    "improvement",
    "invariance",
    "calibration",
    "phase",
];

/// Scan `runs_dir` for `*/metrics.json`, validate provenance against the
/// current repo state, evaluate every recognized gate against each run's
/// metrics, and write `results/RESULTS.md` + `results/RESULTS.json`.
///
/// Refuses (returns `Err`, writes nothing) if ANY run's `git_sha` doesn't
/// match the current `HEAD`, or its `code_fingerprint` doesn't match the
/// current source tree, or (for a real dataset, i.e. not the `"none"`
/// harness-self-check sentinel) its `dataset_sha256` isn't listed in the
/// committed `data/SPLIT_HASHES`.
pub fn generate(cfg: &ReportConfig) -> Result<(), ReportError> {
    let records = load_and_validate_runs(cfg)?;
    let gates = load_gates(cfg.gates_toml).map_err(|e| ReportError::Gate(e.to_string()))?;

    let mut entries = Vec::new();
    for (run_id, record) in records {
        let mut gate_entries = Vec::new();
        for &name in GATE_NAMES {
            match evaluate(name, &gates, &record.metrics) {
                Ok(GateOutcome::Pass {
                    measured,
                    threshold,
                }) => {
                    gate_entries.push(GateEntry {
                        name: name.to_string(),
                        outcome: "PASS".to_string(),
                        measured: Some(measured),
                        threshold: Some(threshold),
                    });
                }
                Ok(GateOutcome::Fail {
                    measured,
                    threshold,
                }) => {
                    gate_entries.push(GateEntry {
                        name: name.to_string(),
                        outcome: "FAIL".to_string(),
                        measured: Some(measured),
                        threshold: Some(threshold),
                    });
                }
                Err(_) => {
                    // No metric for this gate in this run: N/A, not a fabricated PASS/FAIL.
                    gate_entries.push(GateEntry {
                        name: name.to_string(),
                        outcome: "N/A".to_string(),
                        measured: None,
                        threshold: None,
                    });
                }
            }
        }
        entries.push(RunEntry {
            run_id,
            git_sha: record.git_sha,
            dataset_sha256: record.dataset_sha256,
            seed: record.seed,
            backend: record.backend,
            metrics: record.metrics,
            gates: gate_entries,
        });
    }
    // Deterministic ordering (run_id is content-addressed already; sorting
    // here just makes the generated report reproducible byte-for-byte).
    entries.sort_by(|a, b| a.run_id.cmp(&b.run_id));

    fs::create_dir_all(cfg.results_dir)?;

    let results_json = ResultsJson { runs: entries };
    let json_text = serde_json::to_string_pretty(&results_json)
        .map_err(|e| ReportError::Io(std::io::Error::other(e)))?;
    fs::write(cfg.results_dir.join("RESULTS.json"), &json_text)?;

    let md = render_markdown(&results_json);
    fs::write(cfg.results_dir.join("RESULTS.md"), md)?;

    Ok(())
}

/// Regenerate into a scratch directory and diff against the committed
/// `results/`. Returns `Ok(true)` if they match (no drift), `Ok(false)` if
/// they differ (someone hand-edited the committed report, or the metrics
/// actually changed).
///
/// The comparison ignores `git_sha` specifically: a run's `git_sha` is
/// stamped with whatever HEAD was at the moment it was generated, which is
/// necessarily the *parent* of the commit that first checks the resulting
/// `RESULTS.md` in (a file can't know its own future commit hash). Without
/// this exclusion, `check()` would report "drift" on every single commit
/// even when zero metrics changed, which would make the CI gate useless.
/// `git_sha` freshness (T6: stale/fabricated commit) is still enforced
/// separately and strictly by `generate()`'s provenance validation, which
/// this function also runs (via the scratch regeneration) — so a run whose
/// `git_sha` doesn't match the *actual current* HEAD still makes `check()`
/// return `Err`, not just a quiet drift.
pub fn check(cfg: &ReportConfig) -> Result<bool, ReportError> {
    let scratch = tempfile::tempdir()?;
    let scratch_cfg = ReportConfig {
        runs_dir: cfg.runs_dir,
        gates_toml: cfg.gates_toml,
        results_dir: scratch.path(),
        workspace_root: cfg.workspace_root,
    };
    generate(&scratch_cfg)?;

    let committed_json =
        fs::read_to_string(cfg.results_dir.join("RESULTS.json")).unwrap_or_default();
    let fresh_json = fs::read_to_string(scratch.path().join("RESULTS.json"))?;

    let mut committed: serde_json::Value =
        serde_json::from_str(&committed_json).unwrap_or(serde_json::Value::Null);
    let mut fresh: serde_json::Value =
        serde_json::from_str(&fresh_json).map_err(|e| ReportError::Io(std::io::Error::other(e)))?;
    strip_volatile_fields(&mut committed);
    strip_volatile_fields(&mut fresh);

    if committed != fresh {
        return Ok(false);
    }

    // The Markdown is fully derived from the (now-equal, volatile-fields-
    // stripped) JSON, so a hand-edit that only touches the .md (like an
    // appended stray line, or a retyped number that doesn't match the JSON
    // it was supposedly rendered from) is caught by re-rendering from the
    // committed JSON's data and comparing text.
    let committed_md = fs::read_to_string(cfg.results_dir.join("RESULTS.md")).unwrap_or_default();
    let fresh_md = fs::read_to_string(scratch.path().join("RESULTS.md")).unwrap_or_default();
    let committed_md_normalized = strip_git_sha_lines(&committed_md);
    let fresh_md_normalized = strip_git_sha_lines(&fresh_md);

    Ok(committed_md_normalized == fresh_md_normalized)
}

/// Remove the `git_sha` field from every run entry in a `RESULTS.json`
/// value (in place), for the reason documented on `check()`.
fn strip_volatile_fields(value: &mut serde_json::Value) {
    if let Some(runs) = value.get_mut("runs").and_then(|r| r.as_array_mut()) {
        for run in runs {
            if let Some(obj) = run.as_object_mut() {
                obj.remove("git_sha");
            }
        }
    }
}

fn strip_git_sha_lines(md: &str) -> String {
    md.lines()
        .filter(|l| !l.trim_start().starts_with("- git_sha:"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn load_and_validate_runs(cfg: &ReportConfig) -> Result<Vec<(String, RunRecord)>, ReportError> {
    let expected_git_sha = current_git_sha(cfg.workspace_root)
        .map_err(|e| ReportError::Provenance(format!("could not determine current HEAD: {e}")))?;
    let expected_fingerprint = code_fingerprint(cfg.workspace_root)
        .map_err(|e| ReportError::Provenance(format!("could not compute code_fingerprint: {e}")))?;
    let split_hashes = load_split_hashes(cfg.workspace_root);

    let mut out = Vec::new();
    if !cfg.runs_dir.exists() {
        return Ok(out);
    }
    for entry in fs::read_dir(cfg.runs_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let run_id = path.file_name().unwrap().to_string_lossy().to_string();
        let metrics_path = path.join("metrics.json");
        if !metrics_path.exists() {
            continue;
        }
        let text = fs::read_to_string(&metrics_path)?;
        let record: RunRecord = serde_json::from_str(&text)
            .map_err(|e| ReportError::Provenance(format!("{run_id}: invalid metrics.json: {e}")))?;

        if record.git_sha != expected_git_sha {
            return Err(ReportError::Provenance(format!(
                "{run_id}: git_sha {} != current HEAD {} (stale or fabricated run)",
                record.git_sha, expected_git_sha
            )));
        }
        if record.code_fingerprint != expected_fingerprint {
            return Err(ReportError::Provenance(format!(
                "{run_id}: code_fingerprint does not match the current qilm-core/qilm-train source tree"
            )));
        }
        if record.dataset_sha256 != "none" {
            match &split_hashes {
                Some(hashes) if hashes.contains(&record.dataset_sha256) => {}
                Some(_) => {
                    return Err(ReportError::Provenance(format!(
                        "{run_id}: dataset_sha256 {} is not in the committed data/SPLIT_HASHES",
                        record.dataset_sha256
                    )));
                }
                None => {
                    return Err(ReportError::Provenance(format!(
                        "{run_id}: dataset_sha256 is not \"none\" but data/SPLIT_HASHES does not exist -- cannot validate"
                    )));
                }
            }
        }

        out.push((run_id, record));
    }
    Ok(out)
}

fn load_split_hashes(workspace_root: &Path) -> Option<Vec<String>> {
    let path = workspace_root.join("data").join("SPLIT_HASHES");
    let text = fs::read_to_string(path).ok()?;
    Some(
        text.lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .collect(),
    )
}

fn render_markdown(results: &ResultsJson) -> String {
    let mut s = String::new();
    s.push_str("<!-- GENERATED by `qilm-train report` — DO NOT EDIT. Edits are reverted by CI (verify-results). -->\n\n");
    s.push_str("# CCR — Results\n\n");

    if results.runs.is_empty() {
        s.push_str("No runs recorded yet.\n");
        return s;
    }

    for run in &results.runs {
        s.push_str(&format!("## Run `{}`\n\n", run.run_id));
        s.push_str(&format!("- git_sha: `{}`\n", run.git_sha));
        s.push_str(&format!("- dataset_sha256: `{}`\n", run.dataset_sha256));
        s.push_str(&format!("- seed: {}\n", run.seed));
        s.push_str(&format!("- backend: {}\n\n", run.backend));

        s.push_str("### Metrics\n\n");
        if let serde_json::Value::Object(map) = &run.metrics {
            if map.is_empty() {
                s.push_str("(none)\n\n");
            } else {
                for (k, v) in map {
                    s.push_str(&format!("- {k}: {v}\n"));
                }
                s.push('\n');
            }
        }

        s.push_str("### Gates\n\n");
        s.push_str("| gate | outcome | measured | threshold |\n|---|---|---|---|\n");
        for g in &run.gates {
            let measured = g
                .measured
                .map(|m| m.to_string())
                .unwrap_or_else(|| "-".to_string());
            let threshold = g
                .threshold
                .map(|t| t.to_string())
                .unwrap_or_else(|| "-".to_string());
            s.push_str(&format!(
                "| {} | {} | {} | {} |\n",
                g.name, g.outcome, measured, threshold
            ));
        }
        s.push('\n');
    }

    s
}

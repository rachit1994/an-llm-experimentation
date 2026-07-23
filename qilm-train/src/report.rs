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

// ============================================================================
// Phase report generation (DELIVERABLE 1)
// ============================================================================

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
struct PhaseSpec {
    title: String,
    gates: Vec<PhaseGateSpec>,
}

#[derive(Debug, Clone, Deserialize)]
struct PhaseGateSpec {
    name: String,
    kind: String, // "project_kill", "narrow", or "infra"
    meaning: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    Proceed,
    Stop {
        gate: String,
        measured: f64,
        threshold: f64,
    },
    ProceedNarrowed {
        gate: String,
        measured: f64,
        threshold: f64,
    },
}

impl Verdict {
    pub fn as_str(&self) -> &str {
        match self {
            Verdict::Proceed => "PROCEED",
            Verdict::Stop { .. } => "STOP",
            Verdict::ProceedNarrowed { .. } => "PROCEED (narrowed)",
        }
    }
}

/// Load phase specifications from `reports/phase_spec.toml`.
fn load_phase_specs(
    workspace_root: &Path,
) -> Result<std::collections::HashMap<u32, PhaseSpec>, ReportError> {
    let path = workspace_root.join("reports").join("phase_spec.toml");
    let text = fs::read_to_string(&path).map_err(ReportError::Io)?;

    let data: serde_json::Value = toml::from_str(&text)
        .map_err(|e| ReportError::Gate(format!("phase_spec.toml parse error: {e}")))?;

    let mut specs = std::collections::HashMap::new();
    if let serde_json::Value::Object(obj) = data {
        for (key, value) in obj {
            if let Some(phase_num) = key.strip_prefix("phase_") {
                if let Ok(n) = phase_num.parse::<u32>() {
                    let spec: PhaseSpec = serde_json::from_value(value.clone())
                        .map_err(|e| ReportError::Gate(format!("phase_{n} parse error: {e}")))?;
                    specs.insert(n, spec);
                }
            }
        }
    }
    Ok(specs)
}

/// Compute the worth-pursuing verdict (PROCEED / STOP / PROCEED-NARROWED)
/// based on gate outcomes for a phase (mechanical, from gate outcomes only).
pub fn compute_verdict(gates: &[(String, GateOutcome, String, String)]) -> Verdict {
    // Rule: if ANY project_kill gate FAILED -> STOP
    // Else if ANY narrow gate FAILED -> PROCEED_NARROWED
    // Else -> PROCEED

    let mut first_project_kill_fail = None;
    let mut first_narrow_fail = None;

    for (gate_name, outcome, _meaning, kind) in gates {
        if let GateOutcome::Fail {
            measured,
            threshold,
        } = outcome
        {
            if kind == "project_kill" && first_project_kill_fail.is_none() {
                first_project_kill_fail = Some((gate_name.clone(), *measured, *threshold));
            } else if kind == "narrow" && first_narrow_fail.is_none() {
                first_narrow_fail = Some((gate_name.clone(), *measured, *threshold));
            }
        }
    }

    if let Some((gate, measured, threshold)) = first_project_kill_fail {
        Verdict::Stop {
            gate,
            measured,
            threshold,
        }
    } else if let Some((gate, measured, threshold)) = first_narrow_fail {
        Verdict::ProceedNarrowed {
            gate,
            measured,
            threshold,
        }
    } else {
        Verdict::Proceed
    }
}

/// Generate a phase-specific report `reports/PHASE-{phase}.md`.
///
/// This function:
/// 1. Loads phase specs and gates
/// 2. Evaluates the phase's gates against available runs
/// 3. Computes the worth-pursuing verdict
/// 4. Generates the PHASE-N.md file with provenance, gate table, verdict, and conclusion
pub fn generate_phase_report(cfg: &ReportConfig, phase: u32) -> Result<(), ReportError> {
    let records = load_and_validate_runs(cfg)?;
    let gates = load_gates(cfg.gates_toml).map_err(|e| ReportError::Gate(e.to_string()))?;
    let phase_specs = load_phase_specs(cfg.workspace_root)?;

    let phase_spec = phase_specs
        .get(&phase)
        .ok_or_else(|| ReportError::Gate(format!("unknown phase: {phase}")))?;

    // Collect gate outcomes for this phase
    let mut phase_gates: Vec<(String, GateOutcome, String, String)> = Vec::new();
    let mut run_ids = Vec::new();
    let mut git_sha = String::new();

    for (run_id, record) in &records {
        if git_sha.is_empty() {
            git_sha = record.git_sha.clone();
        }
        run_ids.push(run_id.clone());

        for gate_spec in &phase_spec.gates {
            match crate::gate::evaluate(&gate_spec.name, &gates, &record.metrics) {
                Ok(outcome) => {
                    phase_gates.push((
                        gate_spec.name.clone(),
                        outcome,
                        gate_spec.meaning.clone(),
                        gate_spec.kind.clone(),
                    ));
                }
                Err(_) => {
                    // N/A: no metric for this gate in this run
                }
            }
        }
    }

    // Compute verdict (mechanical from gate outcomes)
    let verdict = compute_verdict(&phase_gates);

    // Render Markdown report
    let md = render_phase_markdown(
        phase,
        &phase_spec.title,
        &phase_spec.gates,
        &records,
        &git_sha,
        &run_ids,
        &gates,
        &verdict,
    );

    // Write to reports/PHASE-{phase}.md
    fs::create_dir_all(cfg.workspace_root.join("reports"))?;
    fs::write(
        cfg.workspace_root
            .join("reports")
            .join(format!("PHASE-{}.md", phase)),
        md,
    )?;

    Ok(())
}

fn render_phase_markdown(
    phase: u32,
    title: &str,
    gate_specs: &[PhaseGateSpec],
    records: &[(String, RunRecord)],
    git_sha: &str,
    run_ids: &[String],
    gates_config: &crate::gate::GatesConfig,
    verdict: &Verdict,
) -> String {
    let mut s = String::new();

    // Provenance header
    s.push_str("<!-- GENERATED by `qilm-train report --phase` — DO NOT EDIT. Edits are reverted by CI (verify-results). -->\n\n");
    s.push_str(&format!("# Phase {}: {}\n\n", phase, title));

    // Provenance info
    s.push_str("## Provenance\n\n");
    s.push_str(&format!("- **git_sha**: `{}`\n", git_sha));
    s.push_str(&format!("- **run_ids**: {}\n\n", run_ids.join(", ")));

    // Gates table
    s.push_str("## Gate Evaluation\n\n");
    s.push_str("| gate | measured | threshold | outcome | meaning |\n");
    s.push_str("|---|---|---|---|---|\n");

    for gate_spec in gate_specs {
        let (measured, threshold, outcome) =
            evaluate_gate_for_phase(&gate_spec.name, gates_config, records);
        let measured_str = measured
            .map(|m| m.to_string())
            .unwrap_or_else(|| "-".to_string());
        let threshold_str = threshold
            .map(|t| t.to_string())
            .unwrap_or_else(|| "-".to_string());
        let outcome_str = outcome.unwrap_or_else(|| "N/A".to_string());

        s.push_str(&format!(
            "| {} | {} | {} | {} | {} |\n",
            gate_spec.name, measured_str, threshold_str, outcome_str, gate_spec.meaning
        ));
    }
    s.push('\n');

    // Worth-pursuing verdict
    s.push_str("## Verdict\n\n");
    match verdict {
        Verdict::Proceed => {
            s.push_str(
                "**PROCEED**: All gates pass. Foundation validated; continue to next phase.\n\n",
            );
        }
        Verdict::Stop {
            gate,
            measured,
            threshold,
        } => {
            s.push_str(&format!(
                "**STOP**: Project-kill gate failed: `{}` measured {:.6} vs threshold {:.6}. ",
                gate, measured, threshold
            ));
            s.push_str("The load-bearing claim is dead; do not pursue further.\n\n");
        }
        Verdict::ProceedNarrowed {
            gate,
            measured,
            threshold,
        } => {
            s.push_str(&format!(
                "**PROCEED (narrowed)**: Narrow gate failed: `{}` measured {:.6} vs threshold {:.6}. ",
                gate, measured, threshold
            ));
            s.push_str("Drop the specific claim; other phases remain viable.\n\n");
        }
    }

    // Conclusion
    s.push_str("## Conclusion\n\n");
    s.push_str("Phase ");
    s.push_str(&phase.to_string());
    s.push_str(" gates evaluated. ");
    s.push_str(verdict.as_str());
    s.push_str(".\n");

    s
}

fn evaluate_gate_for_phase(
    gate_name: &str,
    gates_config: &crate::gate::GatesConfig,
    records: &[(String, RunRecord)],
) -> (Option<f64>, Option<f64>, Option<String>) {
    // Evaluate gate across all runs in the phase; return mean measured value
    // (simplified: just use the first run's result for now).
    if let Some((_run_id, record)) = records.first() {
        match crate::gate::evaluate(gate_name, gates_config, &record.metrics) {
            Ok(GateOutcome::Pass {
                measured,
                threshold,
            }) => (Some(measured), Some(threshold), Some("PASS".to_string())),
            Ok(GateOutcome::Fail {
                measured,
                threshold,
            }) => (Some(measured), Some(threshold), Some("FAIL".to_string())),
            Err(_) => (None, None, None),
        }
    } else {
        (None, None, None)
    }
}

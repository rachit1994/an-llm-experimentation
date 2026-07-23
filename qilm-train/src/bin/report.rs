//! report — regenerates results/RESULTS.md from runs/ + gates.toml, checking provenance;
//! refuses to emit fabricated/stale numbers (T0.9). `report --check` is what CI diffs.
//!
//! Thin CLI wrapper: all logic lives in `qilm_train::report` so it's directly
//! unit-testable (see `qilm-train/tests/report_harness.rs`).

use qilm_train::provenance::discover_workspace_root;
use qilm_train::report::{check, generate, ReportConfig};
use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let check_mode = std::env::args().any(|a| a == "--check");
    let workspace_root = discover_workspace_root();
    let runs_dir: PathBuf = workspace_root.join("runs");
    let gates_toml: PathBuf = workspace_root.join("gates.toml");
    let results_dir: PathBuf = workspace_root.join("results");

    let cfg = ReportConfig {
        runs_dir: &runs_dir,
        gates_toml: &gates_toml,
        results_dir: &results_dir,
        workspace_root: &workspace_root,
    };

    if check_mode {
        match check(&cfg) {
            Ok(true) => {
                println!(
                    "report --check: results/ matches a fresh regeneration from runs/ + gates.toml"
                );
                ExitCode::SUCCESS
            }
            Ok(false) => {
                eprintln!("report --check: DRIFT — committed results/ does not match a fresh regeneration");
                ExitCode::FAILURE
            }
            Err(e) => {
                eprintln!("report --check: {e}");
                ExitCode::FAILURE
            }
        }
    } else {
        match generate(&cfg) {
            Ok(()) => {
                println!("report: wrote {}", results_dir.join("RESULTS.md").display());
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("report: refused to generate — {e}");
                ExitCode::FAILURE
            }
        }
    }
}

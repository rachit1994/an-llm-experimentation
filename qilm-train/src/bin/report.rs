//! report — regenerates results/RESULTS.md from runs/ + gates.toml, checking provenance;
//! refuses to emit fabricated/stale numbers (T0.9). `report --check` is what CI diffs.
//! `report --phase N` generates per-phase reports (reports/PHASE-N.md).
//!
//! Thin CLI wrapper: all logic lives in `qilm_train::report` so it's directly
//! unit-testable (see `qilm-train/tests/report_harness.rs`).

use qilm_train::provenance::discover_workspace_root;
use qilm_train::report::{check, generate, generate_phase_report, ReportConfig};
use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let check_mode = args.iter().any(|a| a == "--check");
    let phase_mode = args.iter().position(|a| a == "--phase");

    let workspace_root = discover_workspace_root();
    let runs_dir: PathBuf = workspace_root.join("runs");
    let gates_toml: PathBuf = workspace_root.join("gates.toml");
    let results_dir: PathBuf = workspace_root.join("results");
    let reports_dir: PathBuf = workspace_root.join("reports");

    let cfg = ReportConfig {
        runs_dir: &runs_dir,
        gates_toml: &gates_toml,
        results_dir: &results_dir,
        reports_dir: &reports_dir,
        workspace_root: &workspace_root,
    };

    if let Some(phase_idx) = phase_mode {
        // --phase N mode
        if phase_idx + 1 >= args.len() {
            eprintln!("report --phase: phase number required");
            return ExitCode::FAILURE;
        }
        match args[phase_idx + 1].parse::<u32>() {
            Ok(phase) => match generate_phase_report(&cfg, phase) {
                Ok(()) => {
                    println!(
                        "report --phase {}: wrote {}",
                        phase,
                        workspace_root
                            .join("reports")
                            .join(format!("PHASE-{}.md", phase))
                            .display()
                    );
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("report --phase {}: refused to generate — {e}", phase);
                    ExitCode::FAILURE
                }
            },
            Err(_) => {
                eprintln!("report --phase: invalid phase number");
                ExitCode::FAILURE
            }
        }
    } else if check_mode {
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

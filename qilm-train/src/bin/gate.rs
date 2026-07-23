//! gate <name> <metrics.json> — reads gates.toml, evaluates the named gate
//! against the run's metrics, exits 0 (PASS) / 1 (FAIL) / 2 (error).
//! (T0.9, tests/README.md H3: "gate(name, metrics.json) -> exit 0/1")

use qilm_train::provenance::discover_workspace_root;
use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: gate <name> <metrics.json>");
        return ExitCode::from(2);
    }
    let name = &args[1];
    let metrics_json = PathBuf::from(&args[2]);
    let gates_toml = discover_workspace_root().join("gates.toml");

    let code = qilm_train::gate::gate(name, &gates_toml, &metrics_json);
    ExitCode::from(code as u8)
}

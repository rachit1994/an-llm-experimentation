//! T0.9 — provenance::write_metrics stamps the required fields
//! (VERIFICATION.md §4.1) and produces a file that round-trips through JSON.

use qilm_train::provenance::{write_metrics, MetricsInput, RunRecord};
use serde_json::json;

#[test]
fn write_metrics_stamps_required_fields() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let runs_dir = tmp.path();

    let input = MetricsInput {
        config_sha256: "abc123".to_string(),
        dataset_sha256: "none".to_string(),
        seed: 0,
        backend: "cpu-scalar-oracle".to_string(),
        wall_clock_s: 1.5,
        metrics: json!({ "example_metric": 0.42 }),
    };

    let path = write_metrics(runs_dir, "test-run", input).expect("write_metrics should succeed");
    assert!(path.exists(), "metrics.json should exist at {path:?}");

    let contents = std::fs::read_to_string(&path).unwrap();
    let record: RunRecord = serde_json::from_str(&contents)
        .expect("metrics.json must be valid JSON matching RunRecord");

    assert!(!record.git_sha.is_empty(), "git_sha must be stamped");
    assert_eq!(record.config_sha256, "abc123");
    assert_eq!(record.dataset_sha256, "none");
    assert!(
        !record.code_fingerprint.is_empty(),
        "code_fingerprint must be stamped"
    );
    assert_eq!(record.seed, 0);
    assert_eq!(record.backend, "cpu-scalar-oracle");
    assert_eq!(record.wall_clock_s, 1.5);
    assert!(!record.host.is_empty(), "host must be stamped");
    assert_eq!(record.metrics["example_metric"], 0.42);
}

#[test]
fn write_metrics_git_sha_matches_actual_head() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let input = MetricsInput {
        config_sha256: "x".to_string(),
        dataset_sha256: "none".to_string(),
        seed: 0,
        backend: "cpu".to_string(),
        wall_clock_s: 0.0,
        metrics: json!({}),
    };
    let path = write_metrics(tmp.path(), "run", input).unwrap();
    let record: RunRecord = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();

    // Independently compute HEAD via `git rev-parse HEAD` in the workspace
    // root, so this test doesn't just compare the kernel's output to itself.
    let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap();
    let out = std::process::Command::new("git")
        .arg("rev-parse")
        .arg("HEAD")
        .current_dir(workspace_root)
        .output()
        .expect("git must be available");
    let expected = String::from_utf8_lossy(&out.stdout).trim().to_string();

    assert_eq!(record.git_sha, expected);
}

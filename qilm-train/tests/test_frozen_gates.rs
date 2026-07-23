//! T0.9 — test_frozen_gates: sha256(gates.toml) == gates.lock (VERIFICATION §4.3).
//! Never edit gates.toml to make a run pass; a threshold change must be a
//! visible, reviewable act that updates gates.lock in the same commit.

use qilm_train::gate::gates_lock_matches;
use std::path::Path;

#[test]
fn test_frozen_gates() {
    // Workspace root is the parent of this crate's manifest dir.
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("qilm-train has a parent workspace dir");
    let gates_toml = workspace_root.join("gates.toml");
    let gates_lock = workspace_root.join("gates.lock");

    let matches =
        gates_lock_matches(&gates_toml, &gates_lock).expect("failed to read gates.toml/gates.lock");
    assert!(
        matches,
        "sha256(gates.toml) does not match the committed gates.lock"
    );
}

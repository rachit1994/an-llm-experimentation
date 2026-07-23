//! provenance — signed run artifacts (T0.9, VERIFICATION.md §4.1).
//!
//! Every run writes `runs/<run_id>/metrics.json`, stamped with `git_sha`,
//! `code_fingerprint`, `host`, and the caller-supplied `config_sha256` /
//! `dataset_sha256` / `seed` / `backend` / `wall_clock_s` / `metrics`. This is
//! the ONLY way a number is allowed to enter `results/RESULTS.md` — the
//! report generator (report.rs) reads these files, recomputes, and refuses to
//! emit anything it can't trace back to a `metrics.json` with matching
//! provenance.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Hex-encode bytes (avoids depending on a `LowerHex` impl for whatever
/// fixed-size array type this `sha2` version returns from `finalize()`).
fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// A fully-stamped run artifact, matching the schema in VERIFICATION.md §4.1.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunRecord {
    pub git_sha: String,
    pub config_sha256: String,
    /// The sha256 of the dataset actually loaded, or the literal sentinel
    /// `"none"` for runs that touch no dataset at all (Phase 0's harness
    /// self-checks). A headline model run must NEVER use `"none"` — `report`
    /// checks any other value against the committed `data/SPLIT_HASHES`.
    pub dataset_sha256: String,
    pub code_fingerprint: String,
    pub seed: u64,
    pub backend: String,
    pub wall_clock_s: f64,
    pub host: String,
    pub metrics: serde_json::Value,
}

/// The caller-supplied half of a run record; `write_metrics` fills in
/// `git_sha`, `code_fingerprint`, and `host` itself so a caller can never
/// fabricate those three fields.
pub struct MetricsInput {
    pub config_sha256: String,
    pub dataset_sha256: String,
    pub seed: u64,
    pub backend: String,
    pub wall_clock_s: f64,
    pub metrics: serde_json::Value,
}

/// Write `<runs_dir>/<run_id>/metrics.json`, stamping provenance. Returns the
/// path written.
pub fn write_metrics(runs_dir: &Path, run_id: &str, input: MetricsInput) -> io::Result<PathBuf> {
    let workspace_root = discover_workspace_root();

    let record = RunRecord {
        git_sha: current_git_sha(&workspace_root)?,
        config_sha256: input.config_sha256,
        dataset_sha256: input.dataset_sha256,
        code_fingerprint: code_fingerprint(&workspace_root)?,
        seed: input.seed,
        backend: input.backend,
        wall_clock_s: input.wall_clock_s,
        host: hostname(),
        metrics: input.metrics,
    };

    let dir = runs_dir.join(run_id);
    fs::create_dir_all(&dir)?;
    let path = dir.join("metrics.json");
    let json = serde_json::to_string_pretty(&record)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    fs::write(&path, json)?;
    Ok(path)
}

/// The workspace root: this crate's manifest directory's parent (the
/// `[workspace]` root defined in the top-level `Cargo.toml`). Resolved at
/// compile time via `CARGO_MANIFEST_DIR`, so it does not depend on the
/// current working directory of whatever process links this crate.
pub fn discover_workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("qilm-train's manifest dir must have a parent (the workspace root)")
        .to_path_buf()
}

/// `git rev-parse HEAD`, run in `workspace_root`.
pub fn current_git_sha(workspace_root: &Path) -> io::Result<String> {
    let out = Command::new("git")
        .arg("rev-parse")
        .arg("HEAD")
        .current_dir(workspace_root)
        .output()?;
    if !out.status.success() {
        return Err(io::Error::other(format!(
            "git rev-parse HEAD failed: {}",
            String::from_utf8_lossy(&out.stderr)
        )));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// sha256 over the concatenated (sorted, for determinism) source files of
/// `qilm-core` and `qilm-train` — a coarse but honest fingerprint of "the
/// code that produced this number."
pub fn code_fingerprint(workspace_root: &Path) -> io::Result<String> {
    let mut files: Vec<PathBuf> = Vec::new();
    for crate_name in ["qilm-core", "qilm-train"] {
        let src = workspace_root.join(crate_name).join("src");
        collect_rs_files(&src, &mut files)?;
    }
    files.sort();

    let mut hasher = Sha256::new();
    for f in files {
        let bytes = fs::read(&f)?;
        hasher.update(f.to_string_lossy().as_bytes());
        hasher.update(&bytes);
    }
    Ok(to_hex(&hasher.finalize()))
}

fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
    if !dir.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, out)?;
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
    Ok(())
}

/// Best-effort hostname (metadata only — nc_determinism explicitly excludes
/// `host` from the reproducibility comparison, so this never needs to be
/// deterministic, only present).
pub fn hostname() -> String {
    if let Ok(h) = std::env::var("HOSTNAME") {
        if !h.is_empty() {
            return h;
        }
    }
    if let Ok(out) = Command::new("hostname").output() {
        if out.status.success() {
            let h = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !h.is_empty() {
                return h;
            }
        }
    }
    "unknown-host".to_string()
}

/// sha256 of a byte slice, hex-encoded — used by callers to compute
/// `config_sha256` / `dataset_sha256` consistently.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    to_hex(&hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_hex_matches_known_value() {
        // sha256("") is the well-known empty-string digest (independently
        // known, not computed by this function).
        let expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        assert_eq!(sha256_hex(b""), expected);
    }
}

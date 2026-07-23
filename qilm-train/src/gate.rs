//! gate — evaluate a named gate from `gates.toml` against a `metrics.json`
//! (T0.9). Gates are code: the threshold and the comparison live here, never
//! typed by hand into a report.

use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::fs;
use std::io;
use std::path::Path;

/// Hex-encode bytes (see provenance.rs's copy of this helper for why: this
/// `sha2` version's `finalize()` output doesn't implement `LowerHex`).
fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Mirrors `gates.toml` (VERIFICATION.md §4.3 / METRICS-AND-GATES.md §7)
/// field-for-field. Deliberately does NOT implement `Default` — every field
/// must come from the committed, hash-frozen file.
#[derive(Debug, Clone, Deserialize)]
pub struct GatesConfig {
    pub gradcheck: GradcheckGate,
    pub collapse: CollapseGate,
    pub g0: G0Gate,
    pub improvement: ImprovementGate,
    pub invariance: InvarianceGate,
    pub calibration: CalibrationGate,
    pub phase: PhaseGate,
    pub stats: StatsGate,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GradcheckGate {
    pub max_rel_err: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CollapseGate {
    pub erank_ratio_min: f64,
    pub meanstd_ratio_min: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct G0Gate {
    pub bpb_ratio_max: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ImprovementGate {
    pub param_efficiency_iso_dl: f64,
    pub bpb_gap_isoparam_max: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct InvarianceGate {
    pub within_min: f64,
    pub between_max: f64,
    pub margin_min: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CalibrationGate {
    pub ece_bins: u32,
    pub ece_delta_min: f64,
    pub accuracy_guard_pt: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PhaseGate {
    pub delta_acc_pt: f64,
    pub delta_bpb_rel: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StatsGate {
    pub seeds: u32,
    pub alpha: f64,
}

/// The outcome of evaluating one named gate against one metrics blob.
#[derive(Debug, Clone, PartialEq)]
pub enum GateOutcome {
    Pass { measured: f64, threshold: f64 },
    Fail { measured: f64, threshold: f64 },
}

impl GateOutcome {
    pub fn passed(&self) -> bool {
        matches!(self, GateOutcome::Pass { .. })
    }
}

#[derive(Debug)]
pub enum GateError {
    Io(io::Error),
    Toml(String),
    UnknownGate(String),
    MissingMetric(String),
}

impl std::fmt::Display for GateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GateError::Io(e) => write!(f, "io error: {e}"),
            GateError::Toml(e) => write!(f, "gates.toml parse error: {e}"),
            GateError::UnknownGate(name) => write!(f, "unknown gate: {name}"),
            GateError::MissingMetric(name) => {
                write!(f, "metrics.json missing required field: {name}")
            }
        }
    }
}
impl std::error::Error for GateError {}
impl From<io::Error> for GateError {
    fn from(e: io::Error) -> Self {
        GateError::Io(e)
    }
}

/// Load and parse `gates.toml`.
pub fn load_gates(gates_toml: &Path) -> Result<GatesConfig, GateError> {
    let text = fs::read_to_string(gates_toml)?;
    toml::from_str(&text).map_err(|e| GateError::Toml(e.to_string()))
}

/// sha256 of `gates.toml`'s raw bytes, hex-encoded.
pub fn gates_sha256(gates_toml: &Path) -> io::Result<String> {
    let bytes = fs::read(gates_toml)?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(to_hex(&hasher.finalize()))
}

/// True iff `sha256(gates_toml) == gates_lock` (trimmed). This is the
/// `frozen-gates` CI check made a library function so it's testable directly.
pub fn gates_lock_matches(gates_toml: &Path, gates_lock: &Path) -> io::Result<bool> {
    let computed = gates_sha256(gates_toml)?;
    let locked = fs::read_to_string(gates_lock)?;
    Ok(computed == locked.trim())
}

/// Evaluate a named gate against the `"metrics"` object of a `RunRecord` (see
/// provenance.rs). Implemented so far:
///   - `gradcheck` (Phase 0): `gradcheck_max_rel_err < [gradcheck].max_rel_err`.
///   - `collapse` (H3): `erank_ratio` and `meanstd_ratio` both clear their mins.
///   - `g0` (H2): `bpb_ratio <= [g0].bpb_ratio_max`.
///
/// The remaining names (`improvement`, `invariance`, `calibration`, `phase`)
/// are recognized so the dispatch/threshold plumbing exists, but return
/// `MissingMetric` until their phase produces the field — `gate()` never
/// fabricates a PASS/FAIL for a metric that isn't actually present.
pub fn evaluate(
    name: &str,
    gates: &GatesConfig,
    metrics: &serde_json::Value,
) -> Result<GateOutcome, GateError> {
    match name {
        "gradcheck" => {
            let measured = metrics
                .get("gradcheck_max_rel_err")
                .and_then(|v| v.as_f64())
                .ok_or_else(|| GateError::MissingMetric("gradcheck_max_rel_err".to_string()))?;
            let threshold = gates.gradcheck.max_rel_err;
            Ok(if measured < threshold {
                GateOutcome::Pass {
                    measured,
                    threshold,
                }
            } else {
                GateOutcome::Fail {
                    measured,
                    threshold,
                }
            })
        }
        "collapse" => {
            // H3 anti-collapse: PASS iff BOTH ratios clear their frozen mins.
            // GateOutcome carries a single (measured, threshold), so report the
            // BINDING sub-constraint — the ratio with the smaller margin to its
            // min — which is the one a reader must watch. Both are `>=` gates.
            let erank_ratio = metrics
                .get("erank_ratio")
                .and_then(|v| v.as_f64())
                .ok_or_else(|| GateError::MissingMetric("erank_ratio".to_string()))?;
            let meanstd_ratio = metrics
                .get("meanstd_ratio")
                .and_then(|v| v.as_f64())
                .ok_or_else(|| GateError::MissingMetric("meanstd_ratio".to_string()))?;
            let erank_min = gates.collapse.erank_ratio_min;
            let meanstd_min = gates.collapse.meanstd_ratio_min;
            let passed = erank_ratio >= erank_min && meanstd_ratio >= meanstd_min;
            // Binding sub-constraint = smaller margin (ratio - its min).
            let (measured, threshold) =
                if (erank_ratio - erank_min) <= (meanstd_ratio - meanstd_min) {
                    (erank_ratio, erank_min)
                } else {
                    (meanstd_ratio, meanstd_min)
                };
            Ok(if passed {
                GateOutcome::Pass {
                    measured,
                    threshold,
                }
            } else {
                GateOutcome::Fail {
                    measured,
                    threshold,
                }
            })
        }
        "g0" => {
            // H2 feasibility: PASS iff bpb_ratio <= frozen max (a `<=` gate).
            let bpb_ratio = metrics
                .get("bpb_ratio")
                .and_then(|v| v.as_f64())
                .ok_or_else(|| GateError::MissingMetric("bpb_ratio".to_string()))?;
            let threshold = gates.g0.bpb_ratio_max;
            Ok(if bpb_ratio <= threshold {
                GateOutcome::Pass {
                    measured: bpb_ratio,
                    threshold,
                }
            } else {
                GateOutcome::Fail {
                    measured: bpb_ratio,
                    threshold,
                }
            })
        }
        "improvement" | "invariance" | "calibration" | "phase" => {
            // Recognized gate families with no metric yet (their phases come
            // later). Explicitly distinct from "unknown gate" so callers can
            // tell "not implemented yet" from "typo".
            Err(GateError::MissingMetric(format!(
                "gate '{name}' recognized but no metrics.json field for it exists yet"
            )))
        }
        other => Err(GateError::UnknownGate(other.to_string())),
    }
}

/// CLI-style entry point: exit 0 for PASS, 1 for FAIL, 2 for error. Kept as a
/// thin wrapper so `evaluate` stays the unit-testable core.
pub fn gate(name: &str, gates_toml: &Path, metrics_json: &Path) -> i32 {
    let gates = match load_gates(gates_toml) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("gate: {e}");
            return 2;
        }
    };
    let text = match fs::read_to_string(metrics_json) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("gate: {e}");
            return 2;
        }
    };
    let record: serde_json::Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("gate: invalid metrics.json: {e}");
            return 2;
        }
    };
    let metrics = record
        .get("metrics")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    match evaluate(name, &gates, &metrics) {
        Ok(GateOutcome::Pass {
            measured,
            threshold,
        }) => {
            println!("PASS: {name} measured={measured} threshold={threshold}");
            0
        }
        Ok(GateOutcome::Fail {
            measured,
            threshold,
        }) => {
            println!("FAIL: {name} measured={measured} threshold={threshold}");
            1
        }
        Err(e) => {
            eprintln!("gate: {e}");
            2
        }
    }
}

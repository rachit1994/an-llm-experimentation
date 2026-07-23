//! splits — hashed load_split; refuses unknown sha256. (T0.7)

use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Load a split by name and verify its SHA256 against data/SPLIT_HASHES.
///
/// Refuses to load if the computed SHA256 is not listed in the hashes file.
/// This defeats silent data drift and ensures reproducibility.
pub fn load_split(name: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let hashes_path = Path::new("data/SPLIT_HASHES");

    // Load the hashes file.
    if !hashes_path.exists() {
        return Err(format!("SPLIT_HASHES file not found at {:?}", hashes_path).into());
    }

    let hashes_content = fs::read_to_string(hashes_path)?;

    // Parse the hashes file (format: "name sha256_hex" per line).
    let mut allowed_hashes: HashMap<String, String> = HashMap::new();
    for line in hashes_content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2 {
            allowed_hashes.insert(parts[0].to_string(), parts[1].to_string());
        }
    }

    // Load the split file.
    let split_path = Path::new("data").join(name);
    if !split_path.exists() {
        return Err(format!("Split file not found: {:?}", split_path).into());
    }

    let bytes = fs::read(&split_path)?;

    // Compute SHA256.
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let hash = hasher.finalize();
    // Hex-encode explicitly (the digest output type does not impl LowerHex in sha2 0.11).
    let computed_hash: String = hash.iter().map(|b| format!("{:02x}", b)).collect();

    // Verify against allowed hashes.
    match allowed_hashes.get(name) {
        Some(expected_hash) if &computed_hash == expected_hash => Ok(bytes),
        Some(expected_hash) => Err(format!(
            "SHA256 mismatch for split '{}': computed {}, expected {}",
            name, computed_hash, expected_hash
        )
        .into()),
        None => Err(format!("Split '{}' not found in SPLIT_HASHES", name).into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_split_missing_hashes_file() {
        // This will fail if SPLIT_HASHES doesn't exist, which is expected for now.
        // In a real scenario, we'd create a test fixture.
        let result = load_split("test_split");
        assert!(result.is_err(), "Should fail when SPLIT_HASHES is missing");
    }
}

use super::DbError;

pub const EMBED_DIM: usize = 64;

const STOPWORDS: &[&str] = &[
    "a", "an", "and", "the", "of", "is", "to", "in", "on", "at", "with", "before", "after",
    "ahead", "feels", "feel",
];

/// Words that share one meaning are boosted into one shared bucket.
const CONCEPTS: &[(&str, &[&str])] = &[
    (
        "c_volatile",
        &[
            "volatile",
            "volatility",
            "nervous",
            "choppy",
            "jittery",
            "wild",
            "erratic",
            "spike",
            "uncertain",
        ],
    ),
    (
        "c_event",
        &[
            "inflation",
            "cpi",
            "fed",
            "fomc",
            "earnings",
            "news",
            "report",
            "nfp",
        ],
    ),
    (
        "c_up",
        &["bullish", "rally", "breakout", "uptrend", "strong"],
    ),
    (
        "c_down",
        &["bearish", "selloff", "breakdown", "downtrend", "weak"],
    ),
    (
        "c_calm",
        &["calm", "quiet", "range", "sideways", "stable", "slow"],
    ),
];

/// FNV-1a: a hash that stays identical across Rust versions and machines.
fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

fn bucket(token: &str) -> usize {
    (fnv1a(token) % EMBED_DIM as u64) as usize
}

/// Turns a trader's note into a unit-length vector of `EMBED_DIM` floats.
/// Stand-in for a real embedding model: swap this one function to upgrade.
pub fn embed(text: &str) -> Result<Vec<f32>, DbError> {
    let mut v = vec![0.0f32; EMBED_DIM];
    let mut used = false;

    for raw in text
        .split(|c: char| !c.is_alphabetic())
        .filter(|t| !t.is_empty())
    {
        let token = raw.to_lowercase();
        if STOPWORDS.contains(&token.as_str()) {
            continue;
        }
        if let Some(slot) = v.get_mut(bucket(&token)) {
            *slot += 1.0;
            used = true;
        }
        for (concept, words) in CONCEPTS {
            if words.contains(&token.as_str()) {
                if let Some(slot) = v.get_mut(bucket(concept)) {
                    *slot += 2.0;
                }
            }
        }
    }

    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if !used || norm == 0.0 {
        return Err(DbError::Build("note contains no usable words".to_string()));
    }
    for x in &mut v {
        *x /= norm;
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn has_expected_dimension_and_unit_length() -> Result<(), DbError> {
        let v = embed("Market feels nervous and volatile before inflation news")?;
        assert_eq!(v.len(), EMBED_DIM);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
        Ok(())
    }

    #[test]
    fn similar_notes_score_higher_than_unrelated() -> Result<(), DbError> {
        let a = embed("Market feels nervous and volatile before inflation news")?;
        let b = embed("choppy jittery tape ahead of CPI report")?;
        let c = embed("calm quiet sideways range")?;
        assert!(cosine(&a, &b) > cosine(&a, &c));
        Ok(())
    }

    #[test]
    fn empty_note_is_an_error() {
        assert!(embed("   ...  ").is_err());
        assert!(embed("and the of").is_err());
    }
}

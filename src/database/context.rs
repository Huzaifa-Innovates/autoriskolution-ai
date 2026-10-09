use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::embedding::embed;
use super::queries::pattern_context;
use super::{DbError, Store};

/// How many nearest past trades the vector step retrieves.
pub const SIMILAR_K: usize = 5;
/// If even the closest past note is farther than this, we have no real evidence.
pub const MAX_DISTANCE: f64 = 0.35;
/// Win rate used when there is no usable history.
pub const NEUTRAL_WIN_RATE: f64 = 0.5;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SimilarTrade {
    #[serde(rename = "$id")]
    pub id: u64,
    pub note: String,
    pub outcome: String,
    #[serde(rename = "rMultiple")]
    pub r_multiple: f64,
    #[serde(rename = "$distance")]
    pub distance: f64,
}

#[derive(Debug, Deserialize)]
struct PatternName {
    name: String,
}

#[derive(Debug, Deserialize)]
struct RawContext {
    nearest: Vec<SimilarTrade>,
    #[serde(rename = "patternNames")]
    pattern_names: Vec<PatternName>,
    total: u64,
    wins: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct PatternContext {
    pub similar: Vec<SimilarTrade>,
    pub patterns: Vec<String>,
    pub sample_size: u64,
    pub wins: u64,
    /// wins / total, or None when there is no history at all.
    pub raw_win_rate: Option<f64>,
    /// Laplace-smoothed rate that the risk engine should use.
    pub win_rate: f64,
    /// False when matches were too distant or the pattern had no history.
    pub has_evidence: bool,
}

/// Laplace smoothing: (wins + 1) / (total + 2).
pub fn smoothed_win_rate(wins: u64, total: u64) -> f64 {
    (wins as f64 + 1.0) / (total as f64 + 2.0)
}

impl PatternContext {
    pub fn from_response(response: Value) -> Result<Self, DbError> {
        let raw: RawContext = serde_json::from_value(response)
            .map_err(|e| DbError::Request(format!("unexpected response shape: {e}")))?;

        if raw.wins > raw.total {
            return Err(DbError::Request(format!(
                "inconsistent counts: {} wins out of {} trades",
                raw.wins, raw.total
            )));
        }

        let close_enough = raw
            .nearest
            .first()
            .is_some_and(|best| best.distance <= MAX_DISTANCE);
        let has_evidence = close_enough && raw.total > 0;

        let raw_win_rate = (raw.total > 0).then(|| raw.wins as f64 / raw.total as f64);
        let win_rate = if has_evidence {
            smoothed_win_rate(raw.wins, raw.total)
        } else {
            NEUTRAL_WIN_RATE
        };

        Ok(Self {
            similar: raw.nearest,
            patterns: raw.pattern_names.into_iter().map(|p| p.name).collect(),
            sample_size: raw.total,
            wins: raw.wins,
            raw_win_rate,
            win_rate,
            has_evidence,
        })
    }
}

impl Store {
    /// Note text in, historical pattern context out (one hybrid query).
    pub async fn context_for(&self, note: &str) -> Result<PatternContext, DbError> {
        let embedding = embed(note)?;
        let response = self.run(pattern_context(embedding, SIMILAR_K)).await?;
        PatternContext::from_response(response)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::super::HELIX_URL;
    use super::*;

    fn response(distance: f64, total: u64, wins: u64) -> Value {
        json!({
            "nearest": [{
                "$distance": distance, "$id": 4,
                "note": "Choppy erratic price action ahead of Fed news",
                "outcome": "loss", "rMultiple": -1.0
            }],
            "patternNames": [{ "$id": 0, "name": "high_volatility_news" }],
            "total": total,
            "wins": wins
        })
    }

    #[test]
    fn smoothing_pulls_small_samples_toward_half() {
        assert!((smoothed_win_rate(2, 8) - 0.3).abs() < 1e-12);
        assert!((smoothed_win_rate(0, 0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn close_match_uses_pattern_history() -> Result<(), DbError> {
        let ctx = PatternContext::from_response(response(0.05, 8, 2))?;
        assert!(ctx.has_evidence);
        assert_eq!(ctx.patterns, vec!["high_volatility_news".to_string()]);
        assert!((ctx.win_rate - 0.3).abs() < 1e-12);
        assert_eq!(ctx.raw_win_rate, Some(0.25));
        Ok(())
    }

    #[test]
    fn distant_match_falls_back_to_neutral() -> Result<(), DbError> {
        let ctx = PatternContext::from_response(response(0.6, 8, 2))?;
        assert!(!ctx.has_evidence);
        assert!((ctx.win_rate - NEUTRAL_WIN_RATE).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn inconsistent_counts_are_rejected() {
        assert!(PatternContext::from_response(response(0.05, 3, 5)).is_err());
        assert!(PatternContext::from_response(json!({ "oops": 1 })).is_err());
    }

    #[tokio::test]
    #[ignore = "needs HelixDB running with seeded data"]
    async fn live_context_separates_volatile_from_calm() -> Result<(), DbError> {
        let store = Store::connect(HELIX_URL)?;
        store.init().await?;

        let volatile = store
            .context_for("Market feels nervous and volatile before inflation news")
            .await?;
        println!("volatile: {volatile:#?}");
        assert!(
            volatile
                .patterns
                .iter()
                .any(|p| p == "high_volatility_news")
        );
        assert!(volatile.win_rate < 0.5);

        let calm = store.context_for("calm quiet sideways range").await?;
        println!("calm: {calm:#?}");
        assert!(calm.patterns.iter().any(|p| p == "calm_range"));
        assert!(calm.win_rate > 0.5);
        Ok(())
    }
}

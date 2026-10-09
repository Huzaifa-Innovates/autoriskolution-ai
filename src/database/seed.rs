use std::time::Duration;

use serde_json::Value;

use super::embedding::embed;
use super::queries::{
    NewTradeLog, add_pattern, add_trade_log_with_pattern, count_patterns, create_indexes,
};
use super::{DbError, Store};

const PATTERNS: [&str; 3] = ["high_volatility_news", "calm_range", "trend_breakout"];

struct Seed {
    note: &'static str,
    pattern: &'static str,
    won: bool,
    r: f64,
}

const fn s(note: &'static str, pattern: &'static str, won: bool, r: f64) -> Seed {
    Seed {
        note,
        pattern,
        won,
        r,
    }
}

const V: &str = "high_volatility_news";
const C: &str = "calm_range";
const T: &str = "trend_breakout";

const SEEDS: &[Seed] = &[
    s("Nervous volatile market before CPI report", V, false, -1.0),
    s(
        "Choppy erratic price action ahead of Fed news",
        V,
        false,
        -1.0,
    ),
    s(
        "Wild spikes and uncertain tape before inflation data",
        V,
        false,
        -1.0,
    ),
    s("Jittery volatile session around FOMC news", V, true, 1.2),
    s("Uncertain nervous market into NFP report", V, false, -1.0),
    s("Volatile spikes after earnings news", V, false, -1.0),
    s("Erratic choppy market before CPI", V, true, 0.8),
    s("Nervous choppy tape before Fed report", V, false, -1.0),
    s("Calm quiet sideways range", C, true, 1.5),
    s("Slow stable range session", C, true, 1.0),
    s("Stable slow sideways action", C, true, 1.2),
    s("Calm range with slow volume", C, false, -1.0),
    s("Quiet sideways stable tape", C, true, 1.1),
    s("Slow calm range day", C, true, 1.3),
    s("Sideways quiet range", C, false, -1.0),
    s("Calm stable slow session", C, true, 1.0),
    s("Bullish breakout strong uptrend", T, true, 2.0),
    s("Strong rally breakout continuing", T, true, 1.8),
    s("Bearish breakdown weak downtrend", T, true, 1.5),
    s("Selloff breakdown strong momentum", T, false, -1.0),
    s("Uptrend strong bullish rally", T, true, 2.2),
    s("Weak bearish selloff continuing", T, false, -1.0),
    s("Breakout rally strong volume", T, true, 1.6),
    s("Breakdown weak downtrend", T, false, -1.0),
];

impl Store {
    /// Creates the vector index, then seeds demo history if the graph is empty.
    /// Returns how many trades were inserted (0 if already seeded).
    pub async fn init(&self) -> Result<usize, DbError> {
        self.run(create_indexes()?).await?;

        let existing = self.run(count_patterns()).await?;
        let count = existing
            .get("patterns")
            .and_then(Value::as_u64)
            .ok_or_else(|| DbError::Request("unexpected pattern count response".to_string()))?;
        if count > 0 {
            return Ok(0);
        }

        tokio::time::sleep(Duration::from_secs(2)).await; // let the new index activate
        for name in PATTERNS {
            self.run(add_pattern(name)).await?;
        }
        for seed in SEEDS {
            let log = NewTradeLog {
                note: seed.note,
                embedding: embed(seed.note)?,
                won: seed.won,
                r_multiple: seed.r,
            };
            self.run(add_trade_log_with_pattern(log, seed.pattern))
                .await?;
        }
        Ok(SEEDS.len())
    }
}

#[cfg(test)]
mod tests {
    use super::super::HELIX_URL;
    use super::super::queries::pattern_context;
    use super::*;

    #[tokio::test]
    #[ignore = "needs HelixDB running: helix restart dev"]
    async fn hybrid_query_returns_pattern_win_counts() -> Result<(), DbError> {
        let store = Store::connect(HELIX_URL)?;
        let inserted = store.init().await?;
        println!("seeded {inserted} trades");

        let query = embed("Market feels nervous and volatile before inflation news")?;
        let result = store.run(pattern_context(query, 5)).await?;
        println!("{result:#}");

        let total = result.get("total").and_then(Value::as_u64);
        let wins = result.get("wins").and_then(Value::as_u64);
        match (total, wins) {
            (Some(t), Some(w)) => assert!(t > 0 && w <= t, "bad counts: {w}/{t}"),
            _ => {
                return Err(DbError::Request(
                    "missing total/wins in response".to_string(),
                ));
            }
        }
        Ok(())
    }
}

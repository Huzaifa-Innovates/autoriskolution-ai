pub mod context;
pub mod embedding;
pub mod queries;
pub mod seed;

use std::fmt;

use helix_db::Client;
use helix_db::dsl::prelude::QueryRequest;
use serde_json::Value;

pub const HELIX_URL: &str = "http://localhost:6969";

#[derive(Debug)]
pub enum DbError {
    Build(String),
    Request(String),
}

impl fmt::Display for DbError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Build(msg) => write!(f, "could not build query: {msg}"),
            Self::Request(msg) => write!(f, "database request failed: {msg}"),
        }
    }
}

impl std::error::Error for DbError {}

pub struct Store {
    client: Client,
}

impl Store {
    pub fn connect(url: &str) -> Result<Self, DbError> {
        let client = Client::new(Some(url)).map_err(|e| DbError::Request(e.to_string()))?;
        Ok(Self { client })
    }

    pub async fn run(&self, request: QueryRequest) -> Result<Value, DbError> {
        let response: Value = self
            .client
            .query(request)
            .send()
            .await
            .map_err(|e| DbError::Request(e.to_string()))?;
        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::embedding::embed;
    use super::queries::{NewTradeLog, add_trade_log, create_indexes, similar_logs};
    use super::*;

    #[tokio::test]
    #[ignore = "needs HelixDB running: helix start dev"]
    async fn insert_then_vector_search_round_trip() -> Result<(), DbError> {
        let store = Store::connect(HELIX_URL)?;

        store.run(create_indexes()?).await?;
        tokio::time::sleep(Duration::from_secs(2)).await; // let the index activate

        let notes = [
            (
                "Market nervous and volatile before inflation news",
                false,
                -1.0,
            ),
            ("choppy jittery tape ahead of CPI report", false, -1.0),
            ("calm quiet sideways range, slow session", true, 1.5),
        ];
        for (note, won, r) in notes {
            let log = NewTradeLog {
                note,
                embedding: embed(note)?,
                won,
                r_multiple: r,
            };
            store.run(add_trade_log(log)).await?;
        }

        let query = embed("Market feels nervous and volatile before inflation news")?;
        let result = store.run(similar_logs(query, 3)).await?;
        println!("{result:#}");

        let hits = result.get("hits").and_then(Value::as_array);
        assert!(
            hits.is_some_and(|rows| !rows.is_empty()),
            "no vector hits returned"
        );
        Ok(())
    }
}

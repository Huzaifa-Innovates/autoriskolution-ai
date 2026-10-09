use std::num::NonZeroUsize;

use helix_db::dsl::prelude::*;

use super::DbError;
use super::embedding::EMBED_DIM;

pub const TRADE_LABEL: &str = "TradeLog";
pub const EMBED_PROP: &str = "embedding";

pub struct NewTradeLog<'a> {
    pub note: &'a str,
    pub embedding: Vec<f32>,
    pub won: bool,
    pub r_multiple: f64,
}

/// Creates the cosine vector index on TradeLog.embedding (safe to repeat).
pub fn create_indexes() -> Result<QueryRequest, DbError> {
    let dim = NonZeroUsize::new(EMBED_DIM)
        .ok_or_else(|| DbError::Build("embedding dimension is zero".to_string()))?;

    let batch = write_batch()
        .var_as(
            "index",
            g().create_vector_index_nodes(
                TRADE_LABEL,
                EMBED_PROP,
                dim,
                VectorDistanceMetric::Cosine,
                None::<&str>,
            ),
        )
        .returning(["index"]);

    Ok(QueryRequest::write(batch).with_query_name("create_indexes"))
}

/// Stores one historical trade with its note embedding.
#[cfg(test)]
pub fn add_trade_log(log: NewTradeLog<'_>) -> QueryRequest {
    let outcome = if log.won { "win" } else { "loss" };

    let batch = write_batch()
        .var_as(
            "log",
            g().add_n(
                TRADE_LABEL,
                vec![
                    ("note", PropertyInput::from(log.note)),
                    ("outcome", PropertyInput::from(outcome)),
                    ("rMultiple", PropertyInput::from(log.r_multiple)),
                    (EMBED_PROP, PropertyInput::from(log.embedding)),
                ],
            ),
        )
        .returning(["log"]);

    QueryRequest::write(batch).with_query_name("add_trade_log")
}

/// The `k` past trades whose notes are closest to the given embedding.
#[cfg(test)]
pub fn similar_logs(embedding: Vec<f32>, k: usize) -> QueryRequest {
    let batch = read_batch()
        .var_as(
            "hits",
            g().vector_search_nodes(TRADE_LABEL, EMBED_PROP, embedding, k, None)
                .value_map(Some(vec![
                    "$id",
                    "note",
                    "outcome",
                    "rMultiple",
                    "$distance",
                ])),
        )
        .returning(["hits"]);

    QueryRequest::read(batch).with_query_name("similar_logs")
}

pub const PATTERN_LABEL: &str = "Pattern";
pub const PATTERN_EDGE: &str = "HAS_PATTERN";

pub fn add_pattern(name: &str) -> QueryRequest {
    let batch = write_batch()
        .var_as(
            "pattern",
            g().add_n(PATTERN_LABEL, vec![("name", PropertyInput::from(name))]),
        )
        .returning(["pattern"]);

    QueryRequest::write(batch).with_query_name("add_pattern")
}

pub fn count_patterns() -> QueryRequest {
    let batch = read_batch()
        .var_as("patterns", g().n_with_label(PATTERN_LABEL).count())
        .returning(["patterns"]);

    QueryRequest::read(batch).with_query_name("count_patterns")
}

/// Stores a trade and links it to an existing Pattern, in one transaction.
pub fn add_trade_log_with_pattern(log: NewTradeLog<'_>, pattern: &str) -> QueryRequest {
    let outcome = if log.won { "win" } else { "loss" };

    let batch = write_batch()
        .var_as(
            "pattern",
            g().n_with_label_where(PATTERN_LABEL, SourcePredicate::eq("name", pattern)),
        )
        .var_as(
            "log",
            g().add_n(
                TRADE_LABEL,
                vec![
                    ("note", PropertyInput::from(log.note)),
                    ("outcome", PropertyInput::from(outcome)),
                    ("rMultiple", PropertyInput::from(log.r_multiple)),
                    (EMBED_PROP, PropertyInput::from(log.embedding)),
                ],
            ),
        )
        .var_as(
            "link",
            g().n(NodeRef::var("log")).add_e(
                PATTERN_EDGE,
                NodeRef::var("pattern"),
                vec![("source", "journal")],
            ),
        )
        .returning(["log"]);

    QueryRequest::write(batch).with_query_name("add_trade_log_with_pattern")
}

/// THE HYBRID QUERY: vector search -> graph traversal -> pattern win counts.
pub fn pattern_context(embedding: Vec<f32>, k: usize) -> QueryRequest {
    let batch = read_batch()
        // 1. Vector step: the k most similar past trades.
        .var_as(
            "hits",
            g().vector_search_nodes(TRADE_LABEL, EMBED_PROP, embedding.clone(), k, None),
        )
        .var_as(
            "nearest",
            g().vector_search_nodes(TRADE_LABEL, EMBED_PROP, embedding, k, None)
                .value_map(Some(vec![
                    "$id",
                    "note",
                    "outcome",
                    "rMultiple",
                    "$distance",
                ])),
        )
        // 2. Graph step: which patterns do those trades belong to?
        .var_as(
            "patterns",
            g().n(NodeRef::var("hits")).out(Some(PATTERN_EDGE)).dedup(),
        )
        .var_as(
            "patternNames",
            g().n(NodeRef::var("patterns"))
                .value_map(Some(vec!["$id", "name"])),
        )
        // 3. Graph step: every trade ever tagged with those patterns.
        .var_as(
            "total",
            g().n(NodeRef::var("patterns"))
                .in_(Some(PATTERN_EDGE))
                .dedup()
                .count(),
        )
        .var_as(
            "wins",
            g().n(NodeRef::var("patterns"))
                .in_(Some(PATTERN_EDGE))
                .dedup()
                .where_(Predicate::eq("outcome", "win"))
                .count(),
        )
        .returning(["nearest", "patternNames", "total", "wins"]);

    QueryRequest::read(batch).with_query_name("pattern_context")
}

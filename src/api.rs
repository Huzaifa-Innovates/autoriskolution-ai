use std::sync::Arc;

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use tower_http::services::ServeDir;

use crate::database::context::PatternContext;
use crate::database::{DbError, Store};
use crate::risk::RiskError;
use crate::risk::kelly::{fractional_kelly, full_kelly, reward_to_risk};
use crate::risk::monte_carlo::{self, DEFAULT_ITERATIONS, SimConfig, SimResult};

/// Quarter-Kelly: full Kelly is too aggressive for real accounts.
const KELLY_MULTIPLIER: f64 = 0.25;
/// Hard ceiling on risk per trade, whatever Kelly says.
const KELLY_CAP: f64 = 0.05;
/// With no matching history, size like a cautious beginner.
const NO_EVIDENCE_CAP: f64 = 0.01;
const TRADES_PER_PATH: usize = 100;
/// A simulated path is "ruined" if equity falls to 50% of the starting balance.
const RUIN_THRESHOLD: f64 = 0.5;
/// Fixed seed: identical inputs always give identical output.
const SIM_SEED: u64 = 42;
const MAX_NOTE_CHARS: usize = 500;

// ---------- request / response types ----------

#[derive(Debug, Clone, Deserialize)]
pub struct AnalyzeRequest {
    pub balance: f64,
    pub entry: f64,
    pub target: f64,
    pub stop_loss: f64,
    pub note: String,
    /// What the trader was about to risk, in percent of balance (optional).
    pub intended_risk_pct: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    Ok,
    NoEdge,
    OverLeveraged,
    NoEvidence,
}

#[derive(Debug, Serialize)]
pub struct Comparison {
    pub intended_risk_pct: f64,
    pub survival_pct: f64,
    pub median_final_balance: f64,
}

#[derive(Debug, Serialize)]
pub struct AnalyzeResponse {
    pub verdict: Verdict,
    pub reward_to_risk: f64,
    pub win_rate_pct: f64,
    pub full_kelly_pct: f64,
    pub recommended_risk_pct: f64,
    pub risk_amount: f64,
    pub position_units: f64,
    pub survival_pct: f64,
    pub median_final_balance: f64,
    pub p5_final_balance: f64,
    pub p95_final_balance: f64,
    pub comparison: Option<Comparison>,
    pub context: PatternContext,
}

// ---------- errors ----------

#[derive(Debug)]
pub enum ApiError {
    BadRequest(String),
    Upstream(String),
    Internal(String),
}

impl From<RiskError> for ApiError {
    fn from(e: RiskError) -> Self {
        Self::BadRequest(e.to_string())
    }
}

impl From<DbError> for ApiError {
    fn from(e: DbError) -> Self {
        match e {
            DbError::Build(msg) => Self::BadRequest(msg),
            DbError::Request(msg) => Self::Upstream(msg),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            Self::BadRequest(m) => (StatusCode::BAD_REQUEST, m),
            Self::Upstream(m) => (StatusCode::BAD_GATEWAY, m),
            Self::Internal(m) => (StatusCode::INTERNAL_SERVER_ERROR, m),
        };
        (status, Json(serde_json::json!({ "error": message }))).into_response()
    }
}

// ---------- router ----------

pub fn router(store: Arc<Store>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/api/analyze", post(analyze))
        .fallback_service(ServeDir::new("public"))
        .with_state(store)
}

async fn health() -> &'static str {
    "AutoRiskolution AI: online"
}

async fn analyze(
    State(store): State<Arc<Store>>,
    Json(req): Json<AnalyzeRequest>,
) -> Result<Json<AnalyzeResponse>, ApiError> {
    validate(&req)?;
    let context = store.context_for(&req.note).await?;
    let response = assess(req, context).await?;
    Ok(Json(response))
}

// ---------- logic ----------

fn validate(req: &AnalyzeRequest) -> Result<(), ApiError> {
    if !req.balance.is_finite() || req.balance <= 0.0 {
        return Err(ApiError::BadRequest(
            "balance must be a positive number".to_string(),
        ));
    }
    if req.note.trim().is_empty() || req.note.chars().count() > MAX_NOTE_CHARS {
        return Err(ApiError::BadRequest(format!(
            "note must be between 1 and {MAX_NOTE_CHARS} characters"
        )));
    }
    if let Some(pct) = req.intended_risk_pct {
        if !pct.is_finite() || pct <= 0.0 || pct >= 100.0 {
            return Err(ApiError::BadRequest(
                "intended_risk_pct must be between 0 and 100".to_string(),
            ));
        }
    }
    reward_to_risk(req.entry, req.target, req.stop_loss)?;
    Ok(())
}

fn sim_config(balance: f64, win_rate: f64, reward_ratio: f64, risk_fraction: f64) -> SimConfig {
    SimConfig {
        iterations: DEFAULT_ITERATIONS,
        trades_per_path: TRADES_PER_PATH,
        win_rate,
        reward_ratio,
        risk_fraction,
        start_balance: balance,
        ruin_threshold: RUIN_THRESHOLD,
        seed: SIM_SEED,
    }
}

/// Runs the CPU-heavy simulation off the async workers.
async fn simulate(cfg: SimConfig) -> Result<SimResult, ApiError> {
    tokio::task::spawn_blocking(move || monte_carlo::run(&cfg))
        .await
        .map_err(|e| ApiError::Internal(format!("simulation task failed: {e}")))?
        .map_err(ApiError::from)
}

/// Pure calculation step: no database access, so it is easy to test.
pub async fn assess(req: AnalyzeRequest, ctx: PatternContext) -> Result<AnalyzeResponse, ApiError> {
    let ratio = reward_to_risk(req.entry, req.target, req.stop_loss)?;
    let full = full_kelly(ctx.win_rate, ratio)?;
    let cap = if ctx.has_evidence {
        KELLY_CAP
    } else {
        NO_EVIDENCE_CAP
    };
    let fraction = fractional_kelly(ctx.win_rate, ratio, KELLY_MULTIPLIER, cap)?;

    let risk_amount = req.balance * fraction;
    let risk_per_unit = (req.entry - req.stop_loss).abs();
    let position_units = risk_amount / risk_per_unit;

    let sim = simulate(sim_config(req.balance, ctx.win_rate, ratio, fraction)).await?;

    let (comparison, intended_fraction) = match req.intended_risk_pct {
        Some(pct) => {
            let intended = pct / 100.0;
            let alt = simulate(sim_config(req.balance, ctx.win_rate, ratio, intended)).await?;
            let cmp = Comparison {
                intended_risk_pct: pct,
                survival_pct: alt.survival_pct,
                median_final_balance: alt.median_final_balance,
            };
            (Some(cmp), Some(intended))
        }
        None => (None, None),
    };

    let verdict = if fraction <= 0.0 {
        Verdict::NoEdge
    } else if intended_fraction.is_some_and(|i| i > fraction) {
        Verdict::OverLeveraged
    } else {
        if ctx.has_evidence {
            Verdict::Ok
        } else {
            Verdict::NoEvidence
        }
    };

    Ok(AnalyzeResponse {
        verdict,
        reward_to_risk: ratio,
        win_rate_pct: ctx.win_rate * 100.0,
        full_kelly_pct: full * 100.0,
        recommended_risk_pct: fraction * 100.0,
        risk_amount,
        position_units,
        survival_pct: sim.survival_pct,
        median_final_balance: sim.median_final_balance,
        p5_final_balance: sim.p5_final_balance,
        p95_final_balance: sim.p95_final_balance,
        comparison,
        context: ctx,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(win_rate: f64) -> PatternContext {
        PatternContext {
            similar: vec![],
            patterns: vec!["test_pattern".to_string()],
            sample_size: 8,
            wins: 4,
            raw_win_rate: Some(0.5),
            win_rate,
            has_evidence: true,
        }
    }

    fn request(intended: Option<f64>) -> AnalyzeRequest {
        AnalyzeRequest {
            balance: 10_000.0,
            entry: 100.0,
            target: 110.0,
            stop_loss: 95.0,
            note: "test note".to_string(),
            intended_risk_pct: intended,
        }
    }

    #[tokio::test]
    async fn no_edge_means_zero_size() -> Result<(), ApiError> {
        // win rate 0.3, ratio 2.0 -> Kelly = 0.3 - 0.7/2 = -0.05
        let r = assess(request(None), ctx(0.3)).await?;
        assert_eq!(r.verdict, Verdict::NoEdge);
        assert_eq!(r.risk_amount, 0.0);
        assert_eq!(r.position_units, 0.0);
        Ok(())
    }

    #[tokio::test]
    async fn strong_edge_is_capped_and_sized_in_units() -> Result<(), ApiError> {
        // Kelly = 0.7 - 0.3/2 = 0.55; quarter = 0.1375; cap = 0.05
        let r = assess(request(None), ctx(0.7)).await?;
        assert_eq!(r.verdict, Verdict::Ok);
        assert!((r.recommended_risk_pct - 5.0).abs() < 1e-9);
        assert!((r.risk_amount - 500.0).abs() < 1e-9);
        assert!((r.position_units - 100.0).abs() < 1e-9); // $500 / $5 per unit
        assert!(r.survival_pct > 95.0);
        Ok(())
    }

    #[tokio::test]
    async fn over_leveraging_is_flagged_and_less_survivable() -> Result<(), ApiError> {
        let r = assess(request(Some(30.0)), ctx(0.7)).await?;
        assert_eq!(r.verdict, Verdict::OverLeveraged);
        let cmp = r
            .comparison
            .ok_or_else(|| ApiError::Internal("no comparison".to_string()))?;
        assert!(cmp.survival_pct < r.survival_pct);
        Ok(())
    }

    #[test]
    fn bad_input_is_rejected() {
        let mut bad = request(None);
        bad.stop_loss = 105.0; // stop above entry but target also above: same side
        assert!(validate(&bad).is_err());

        let mut empty = request(None);
        empty.note = "   ".to_string();
        assert!(validate(&empty).is_err());

        assert!(validate(&request(Some(150.0))).is_err());
    }
}

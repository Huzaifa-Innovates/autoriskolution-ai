use rand::{Rng, SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use serde::Serialize;

use super::RiskError;

pub const DEFAULT_ITERATIONS: usize = 5_000;

#[derive(Debug, Clone)]
pub struct SimConfig {
    pub iterations: usize,
    pub trades_per_path: usize,
    pub win_rate: f64,
    pub reward_ratio: f64,
    /// Fraction of current equity risked per trade (from Kelly).
    pub risk_fraction: f64,
    pub start_balance: f64,
    /// A path is "ruined" if equity falls to start_balance * ruin_threshold.
    pub ruin_threshold: f64,
    pub seed: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SimResult {
    pub iterations: usize,
    pub survival_pct: f64,
    pub median_final_balance: f64,
    pub p5_final_balance: f64,
    pub p95_final_balance: f64,
}

impl SimConfig {
    fn validate(&self) -> Result<(), RiskError> {
        if self.iterations == 0 || self.trades_per_path == 0 {
            return Err(RiskError::InvalidInput("iterations and trades must be > 0"));
        }
        if !(self.win_rate.is_finite() && (0.0..=1.0).contains(&self.win_rate)) {
            return Err(RiskError::InvalidInput("win rate must be between 0 and 1"));
        }
        if !self.reward_ratio.is_finite() || self.reward_ratio <= 0.0 {
            return Err(RiskError::InvalidInput("reward ratio must be positive"));
        }
        if !(self.risk_fraction.is_finite() && (0.0..1.0).contains(&self.risk_fraction)) {
            return Err(RiskError::InvalidInput("risk fraction must be in [0, 1)"));
        }
        if !self.start_balance.is_finite() || self.start_balance <= 0.0 {
            return Err(RiskError::InvalidInput("balance must be positive"));
        }
        if !(self.ruin_threshold > 0.0 && self.ruin_threshold < 1.0) {
            return Err(RiskError::InvalidInput("ruin threshold must be in (0, 1)"));
        }
        Ok(())
    }
}

/// One simulated future. Returns (final equity, survived?).
fn simulate_path(cfg: &SimConfig, seed: u64) -> (f64, bool) {
    let mut rng = StdRng::seed_from_u64(seed);
    let floor = cfg.start_balance * cfg.ruin_threshold;
    let mut equity = cfg.start_balance;

    for _ in 0..cfg.trades_per_path {
        let stake = equity * cfg.risk_fraction;
        if rng.random::<f64>() < cfg.win_rate {
            equity += stake * cfg.reward_ratio;
        } else {
            equity -= stake;
        }
        if equity <= floor {
            return (equity, false);
        }
    }
    (equity, true)
}

fn percentile(sorted: &[f64], q: f64) -> Result<f64, RiskError> {
    let last = sorted
        .len()
        .checked_sub(1)
        .ok_or(RiskError::EmptySimulation)?;
    let idx = (last as f64 * q).round() as usize;
    sorted.get(idx).copied().ok_or(RiskError::EmptySimulation)
}

/// Runs all paths in parallel across CPU cores. Same seed = same result.
pub fn run(cfg: &SimConfig) -> Result<SimResult, RiskError> {
    cfg.validate()?;

    let outcomes: Vec<(f64, bool)> = (0..cfg.iterations)
        .into_par_iter()
        .map(|i| simulate_path(cfg, cfg.seed.wrapping_add(i as u64)))
        .collect();

    let survivors = outcomes.iter().filter(|(_, ok)| *ok).count();
    let mut finals: Vec<f64> = outcomes.iter().map(|(balance, _)| *balance).collect();
    finals.sort_by(f64::total_cmp);

    Ok(SimResult {
        iterations: outcomes.len(),
        survival_pct: survivors as f64 / outcomes.len() as f64 * 100.0,
        median_final_balance: percentile(&finals, 0.50)?,
        p5_final_balance: percentile(&finals, 0.05)?,
        p95_final_balance: percentile(&finals, 0.95)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(win_rate: f64, reward_ratio: f64, risk_fraction: f64) -> SimConfig {
        SimConfig {
            iterations: DEFAULT_ITERATIONS,
            trades_per_path: 100,
            win_rate,
            reward_ratio,
            risk_fraction,
            start_balance: 10_000.0,
            ruin_threshold: 0.5,
            seed: 42,
        }
    }

    #[test]
    fn same_seed_gives_same_result() -> Result<(), RiskError> {
        let a = run(&cfg(0.55, 1.0, 0.02))?;
        let b = run(&cfg(0.55, 1.0, 0.02))?;
        assert_eq!(a, b);
        Ok(())
    }

    #[test]
    fn small_size_with_edge_survives() -> Result<(), RiskError> {
        let r = run(&cfg(0.55, 1.0, 0.02))?;
        assert!(r.survival_pct > 95.0, "survival was {}", r.survival_pct);
        Ok(())
    }

    #[test]
    fn big_size_without_edge_is_ruined() -> Result<(), RiskError> {
        let r = run(&cfg(0.40, 1.0, 0.20))?;
        assert!(r.survival_pct < 5.0, "survival was {}", r.survival_pct);
        Ok(())
    }

    #[test]
    fn invalid_config_is_an_error() {
        let mut bad = cfg(0.55, 1.0, 0.02);
        bad.iterations = 0;
        assert!(run(&bad).is_err());
    }
}

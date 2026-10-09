use super::RiskError;

fn is_probability(x: f64) -> bool {
    x.is_finite() && (0.0..=1.0).contains(&x)
}

/// Reward-to-risk ratio `b` for a long or short trade.
/// Long: stop < entry < target. Short: target < entry < stop.
pub fn reward_to_risk(entry: f64, target: f64, stop_loss: f64) -> Result<f64, RiskError> {
    if ![entry, target, stop_loss]
        .iter()
        .all(|v| v.is_finite() && *v > 0.0)
    {
        return Err(RiskError::InvalidInput(
            "entry, target and stop loss must be positive numbers",
        ));
    }
    let reward = target - entry;
    let risk = entry - stop_loss;
    if reward * risk <= 0.0 {
        return Err(RiskError::InvalidInput(
            "target and stop loss must sit on opposite sides of entry",
        ));
    }
    Ok(reward.abs() / risk.abs())
}

/// Full Kelly fraction: f* = p - (1 - p) / b. May be negative (no edge).
pub fn full_kelly(win_rate: f64, reward_ratio: f64) -> Result<f64, RiskError> {
    if !is_probability(win_rate) {
        return Err(RiskError::InvalidInput("win rate must be between 0 and 1"));
    }
    if !reward_ratio.is_finite() || reward_ratio <= 0.0 {
        return Err(RiskError::InvalidInput("reward ratio must be positive"));
    }
    Ok(win_rate - (1.0 - win_rate) / reward_ratio)
}

/// Fractional Kelly, floored at 0 (no edge = no trade) and capped at `cap`.
/// Returns the fraction of account balance to risk on this trade.
pub fn fractional_kelly(
    win_rate: f64,
    reward_ratio: f64,
    multiplier: f64,
    cap: f64,
) -> Result<f64, RiskError> {
    if !(multiplier.is_finite() && multiplier > 0.0 && multiplier <= 1.0) {
        return Err(RiskError::InvalidInput("multiplier must be in (0, 1]"));
    }
    if !(cap.is_finite() && cap > 0.0 && cap <= 1.0) {
        return Err(RiskError::InvalidInput("cap must be in (0, 1]"));
    }
    let full = full_kelly(win_rate, reward_ratio)?;
    Ok((full * multiplier).clamp(0.0, cap))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn full_kelly_matches_hand_calculation() -> Result<(), RiskError> {
        // 0.6 - 0.4 / 1.0 = 0.2
        assert!(close(full_kelly(0.6, 1.0)?, 0.2));
        Ok(())
    }

    #[test]
    fn no_edge_gives_zero_size() -> Result<(), RiskError> {
        assert!(close(fractional_kelly(0.4, 1.0, 0.5, 0.25)?, 0.0));
        Ok(())
    }

    #[test]
    fn cap_limits_size() -> Result<(), RiskError> {
        // full = 0.85, half = 0.425, capped to 0.25
        assert!(close(fractional_kelly(0.9, 2.0, 0.5, 0.25)?, 0.25));
        Ok(())
    }

    #[test]
    fn reward_to_risk_long_and_short() -> Result<(), RiskError> {
        assert!(close(reward_to_risk(100.0, 110.0, 95.0)?, 2.0));
        assert!(close(reward_to_risk(100.0, 90.0, 105.0)?, 2.0));
        Ok(())
    }

    #[test]
    fn rejects_bad_inputs() {
        assert!(reward_to_risk(100.0, 90.0, 95.0).is_err());
        assert!(full_kelly(1.5, 1.0).is_err());
        assert!(full_kelly(0.5, 0.0).is_err());
        assert!(fractional_kelly(0.6, 1.0, 0.0, 0.25).is_err());
    }
}

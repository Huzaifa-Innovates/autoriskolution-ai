pub mod kelly;
pub mod monte_carlo;

use std::fmt;

/// Every fallible function in the risk engine returns this error.
#[derive(Debug, Clone, PartialEq)]
pub enum RiskError {
    InvalidInput(&'static str),
    EmptySimulation,
}

impl fmt::Display for RiskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            Self::EmptySimulation => write!(f, "simulation produced no results"),
        }
    }
}

impl std::error::Error for RiskError {}

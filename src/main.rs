mod api;
mod database;
mod risk;

use std::error::Error;
use std::sync::Arc;

use database::{HELIX_URL, Store};
use tokio::net::TcpListener;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let store = Store::connect(HELIX_URL)?;
    let seeded = store
        .init()
        .await
        .map_err(|e| format!("HelixDB is not ready ({e}). Is `helix start dev` running?"))?;
    println!("HelixDB ready ({seeded} demo trades seeded)");

    let app = api::router(Arc::new(store));
    let listener = TcpListener::bind("127.0.0.1:8080").await?;
    println!("AutoRiskolution AI listening on http://127.0.0.1:8080");

    axum::serve(listener, app).await?;
    Ok(())
}

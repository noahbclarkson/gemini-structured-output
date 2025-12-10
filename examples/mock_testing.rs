//! Example: Offline-friendly mocking for unit tests.
//!
//! This demonstrates how to intercept requests with `with_mock` so you can run
//! `cargo test` or examples without network calls or billing.

use gemini_structured_output::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct Contact {
    name: String,
    email: String,
    notes: Vec<String>,
    score: f32,
}

#[tokio::main]
async fn main() -> Result<()> {
    // No API key needed because the mock handler short-circuits the request.
    let client = StructuredClientBuilder::new("mock-key")
        .with_mock(|req: MockRequest| {
            if req.target.contains("Contact") {
                return Ok(r#"{
                        "name": "Mocky McTestface",
                        "email": "mock@example.com",
                        "notes": ["offline", "deterministic", "fast"],
                        "score": 0.99
                    }"#
                .to_string());
            }
            Err(StructuredError::Context(format!(
                "Unexpected mock target: {}",
                req.target
            )))
        })
        .build()?;

    let outcome = client
        .request::<Contact>()
        .system("Extract a contact record from the user text.")
        .user_text("Contact: Mocky McTestface, mock@example.com")
        .execute()
        .await?;

    println!("Received contact (no network calls): {:#?}", outcome.value);
    println!(
        "Mock mode metadata -> network_attempts: {}, parse_attempts: {}",
        outcome.network_attempts, outcome.parse_attempts
    );

    Ok(())
}

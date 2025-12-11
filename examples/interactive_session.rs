use gemini_structured_output::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct ForecastConfig {
    product: String,
    price: f64,
    quantity: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct ForecastOutput {
    revenue: f64,
}

fn compute_output(config: &ForecastConfig) -> ForecastOutput {
    ForecastOutput {
        revenue: config.price * config.quantity as f64,
    }
}

/// End-to-end interactive session showing Q&A, AI-proposed patch, acceptance, and manual change.
#[tokio::main]
async fn main() -> Result<()> {
    let api_key =
        std::env::var("GEMINI_API_KEY").expect("Set GEMINI_API_KEY to run this example with AI");

    let client = StructuredClientBuilder::new(api_key)
        .with_model(Model::Gemini25Flash)
        .build()?;

    let initial_config = ForecastConfig {
        product: "Widget".to_string(),
        price: 10.0,
        quantity: 10,
    };
    let initial_output = compute_output(&initial_config);

    let mut session = InteractiveSession::new(initial_config.clone(), Some(initial_output.clone()));

    // 1) Q&A grounded in current state
    let answer = session
        .chat(
            &client,
            "What is the current product, price, quantity, and revenue?",
        )
        .await?;
    println!("Q&A -> {answer}");

    // 2) Ask AI to propose a config change (staged as pending)
    let pending = session
        .request_change(
            &client,
            "Raise the price by 20% and adjust quantity if needed to maximize revenue.",
        )
        .await?;

    println!(
        "Pending patch from AI:\n{}",
        serde_json::to_string_pretty(&pending.patch)?
    );

    // 3) Accept the pending change and recompute output
    session.accept_change()?;
    let updated_output = compute_output(&session.config);
    session.update_output(Some(updated_output.clone()));
    println!(
        "Accepted config now {:?} with revenue {:.2}",
        session.config, updated_output.revenue
    );

    // 4) Apply a manual override with effect description and automatic output diff
    let mut manual_config = session.config.clone();
    manual_config.quantity += 5; // user decides to bump quantity
    let manual_output = compute_output(&manual_config);
    let effect = ChangeEffect {
        description: format!(
            "Manual increase in quantity raised revenue to {:.2}",
            manual_output.revenue
        ),
        is_positive: Some(true),
    };

    let manual_patch = session.apply_manual_change(manual_config, manual_output, Some(effect))?;
    println!(
        "Manual config diff:\n{}",
        serde_json::to_string_pretty(&manual_patch)?
    );

    // 5) Show rich history kinds for observability
    println!("\nSession history:");
    for entry in &session.history {
        let text = entry
            .message
            .content
            .parts
            .as_ref()
            .and_then(|parts| {
                parts.iter().find_map(|p| {
                    if let gemini_rust::Part::Text { text, .. } = p {
                        Some(text.as_str())
                    } else {
                        None
                    }
                })
            })
            .unwrap_or_default();

        println!("[{}] {:?} :: {}", entry.timestamp, entry.kind, text);
    }

    Ok(())
}

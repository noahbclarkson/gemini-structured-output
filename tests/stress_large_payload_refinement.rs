use gemini_rust::Model;
use gemini_structured_output::prelude::*;
use std::env;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct ScheduleItem {
    id: u32,
    task: String,
    start_hour: u32,
    end_hour: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct DailySchedule {
    day: String,
    items: Vec<ScheduleItem>,
}

#[tokio::test]
async fn test_large_payload_refinement() {
    tracing_subscriber::fmt()
        .with_env_filter("debug")
        .with_test_writer()
        .init();

    let api_key = env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");

    let client = StructuredClientBuilder::new(api_key)
        .with_model(Model::Gemini3Flash)
        .with_default_retries(0)
        .build()
        .expect("Failed to build client");

    // Generate an initial invalid schedule (logically invalid: start > end)
    let initial = DailySchedule {
        day: "Monday".to_string(),
        items: vec![
            ScheduleItem {
                id: 1,
                task: "Wake up".into(),
                start_hour: 7,
                end_hour: 8,
            },
            ScheduleItem {
                id: 2,
                task: "Work".into(),
                start_hour: 17,
                end_hour: 9,
            }, // Invalid!
            ScheduleItem {
                id: 3,
                task: "Sleep".into(),
                start_hour: 22,
                end_hour: 6,
            }, // Invalid wrapping
        ],
    };

    println!("Refining invalid schedule...");
    println!(
        "\nInitial (invalid) schedule:\n{}",
        serde_json::to_string_pretty(&initial).unwrap()
    );

    let result = client
        .refine(
            initial,
            "Fix the schedule. Start time must be strictly before end time. If wrapping over midnight, split into two tasks.",
        )
        .with_validator(|s: &DailySchedule| {
            for item in &s.items {
                if item.start_hour >= item.end_hour {
                    return Some(format!(
                        "Task '{}' has invalid times: start {} >= end {}",
                        item.task, item.start_hour, item.end_hour
                    ));
                }
            }
            None
        })
        .execute()
        .await;

    match result {
        Ok(outcome) => {
            println!("\n✅ Successfully refined schedule!");
            println!("\nRefined Schedule:");
            println!(
                "{}",
                serde_json::to_string_pretty(&outcome.value).unwrap()
            );
            println!("\nRefinement attempts: {}", outcome.attempts.len());

            for item in outcome.value.items {
                assert!(
                    item.start_hour < item.end_hour,
                    "Refinement failed to fix logic for {}",
                    item.task
                );
            }

            println!("\n✅ All assertions passed!");
        }
        Err(e) => {
            eprintln!("\n❌ Request failed with error: {:?}", e);
            eprintln!("\nFull error details: {:#?}", e);
            panic!("Refinement stress test failed: {}", e);
        }
    }
}

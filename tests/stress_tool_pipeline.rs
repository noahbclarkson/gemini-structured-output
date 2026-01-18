use gemini_rust::Model;
use gemini_structured_output::tools::ToolError;
use gemini_structured_output::{gemini_tool, StructuredClientBuilder, ToolRegistry};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::env;

// --- Tools ---

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct DbQuery {
    sql: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct DbRows {
    rows: Vec<String>,
}

#[gemini_tool(description = "Execute a read-only SQL query on the user database.")]
async fn query_db(args: DbQuery) -> Result<DbRows, ToolError> {
    println!("  [Tool] Executing query: {}", args.sql);
    if args.sql.contains("users") {
        Ok(DbRows {
            rows: vec![
                "User:Alice:Active".into(),
                "User:Bob:Inactive".into(),
            ],
        })
    } else if args.sql.contains("orders") {
        Ok(DbRows {
            rows: vec!["Order:101:Alice:$50".into()],
        })
    } else {
        Ok(DbRows { rows: vec![] })
    }
}

// --- Target Structure ---

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct UserAudit {
    username: String,
    status: String,
    total_spend: f64,
    audit_notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct AuditReport {
    audited_users: Vec<UserAudit>,
    total_system_revenue: f64,
}

#[tokio::test]
async fn test_tool_pipeline() {
    tracing_subscriber::fmt()
        .with_env_filter("debug")
        .with_test_writer()
        .init();

    let api_key = env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");

    let client = StructuredClientBuilder::new(api_key)
        .with_model(Model::Gemini3Flash)
        .with_default_retries(0)
        .with_default_tool_steps(10)
        .build()
        .expect("Failed to build client");

    let registry = ToolRegistry::new().register_tool(query_db_tool::registrar());

    let prompt =
        "Audit the users in the database. Join users with their orders to find total spend. Only report on Alice.";

    println!("Sending tool pipeline request to Gemini...");

    let result = client
        .request::<AuditReport>()
        .system("You are an auditor. Use the `query_db` tool to fetch data. Do not guess.")
        .user_text(prompt)
        .with_tools(registry)
        .execute()
        .await;

    match result {
        Ok(outcome) => {
            println!("\n✅ Successfully generated audit report!");
            println!("\nGenerated Audit Report:");
            println!(
                "{}",
                serde_json::to_string_pretty(&outcome.value).unwrap()
            );
            println!("\nTool calls made: {}", outcome.function_calls.len());

            assert_eq!(outcome.value.audited_users.len(), 1);
            let alice = &outcome.value.audited_users[0];
            assert_eq!(alice.username, "Alice");
            assert_eq!(alice.total_spend, 50.0);
            assert!(
                !outcome.function_calls.is_empty(),
                "Model should have used tools"
            );

            println!("\n✅ All assertions passed!");
        }
        Err(e) => {
            eprintln!("\n❌ Request failed with error: {:?}", e);
            eprintln!("\nFull error details: {:#?}", e);
            panic!("Tool pipeline test failed: {}", e);
        }
    }
}

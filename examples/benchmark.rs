//! Minimal benchmarking example for structured outputs.
//! Requires `--features evals`.

use gemini_rust::Model;
use gemini_structured_output::{
    ContextBuilder, EvalSuite, MockRequest, StructuredClientBuilder, StructuredError,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::env;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct IncidentReport {
    severity: String,
    impacted_services: Vec<String>,
    root_cause: String,
    start_time: String,
    end_time: String,
    timeline: Vec<TimelineEvent>,
    actions: Vec<ActionItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct TimelineEvent {
    time: String,
    description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct ActionItem {
    owner: String,
    action: String,
    due: String,
}

#[derive(Debug, Clone)]
struct ExpectedCase {
    impacted_keywords: Vec<&'static str>,
    cause_keywords: Vec<&'static str>,
    owners: Vec<&'static str>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let use_mock = env::var("USE_MOCK").is_ok();
    let api_key = env::var("GEMINI_API_KEY").unwrap_or_else(|_| {
        if use_mock {
            "mock-api-key".to_string()
        } else {
            panic!("GEMINI_API_KEY must be set")
        }
    });
    let model = Model::Custom("models/gemini-2.5-flash-lite-preview-09-2025".to_string());

    let mut builder = StructuredClientBuilder::new(api_key).with_model(model);

    if use_mock {
        builder = builder.with_mock(|req: MockRequest| {
            if req.target.contains("IncidentReport") {
                Ok(mock_incident().to_string())
            } else {
                Err(StructuredError::Context(format!(
                    "Unexpected mock target: {}",
                    req.target
                )))
            }
        });
        println!("Running in mock mode (no network requests)");
    }

    let client = builder.build()?;

    let cases: Vec<(String, (String, ExpectedCase))> = vec![
        (
            "Partial timeline".to_string(),
            (
                "Incident: checkout errors started 9:05 UTC, resolved 9:32. Cart and payments impacted. Cause was a bad feature flag rollout. Actions: rollback flag; add preflight check. Pagerduty owner: Maya.".to_string(),
                ExpectedCase {
                    impacted_keywords: vec!["cart", "payments"],
                    cause_keywords: vec!["feature flag"],
                    owners: vec!["Maya"],
                },
            ),
        ),
        (
            "Noisy log".to_string(),
            (
                "FYI: at 14:10 we saw spike 500s in /api/search. 14:15 disabled ES shard. 14:40 restored. Root cause: mis-sized cluster after autoscale bug. Services: search-api. Mitigation owners: Luis (scale policy fix by Friday), Priya (add alert).".to_string(),
                ExpectedCase {
                    impacted_keywords: vec!["search"],
                    cause_keywords: vec!["autoscale"],
                    owners: vec!["Luis", "Priya"],
                },
            ),
        ),
        (
            "Messy narrative".to_string(),
            (
                "Customers in EU reported login failures this morning. Timeline: 07:00 auth latency, 07:12 cascading timeouts, 07:25 cleared cache, 07:40 redeployed auth. Cause probably stale cert in auth cache. Impact: login, profile. Follow-ups: rotate cert daily (Alice), add cache TTL monitor (Ben).".to_string(),
                ExpectedCase {
                    impacted_keywords: vec!["login", "profile"],
                    cause_keywords: vec!["cert"],
                    owners: vec!["Alice", "Ben"],
                },
            ),
        ),
        (
            "Minimal info".to_string(),
            (
                "We had an outage 18:05-18:45. Only uploads were down. Cause: s3 bucket policy change. Action: revert policy (owner Ken), add deployment checklist (owner Kim).".to_string(),
                ExpectedCase {
                    impacted_keywords: vec!["upload"],
                    cause_keywords: vec!["policy"],
                    owners: vec!["Ken", "Kim"],
                },
            ),
        ),
        (
            "Mixed tense".to_string(),
            (
                "Next week we will fix the alert gap, but today 10:00-10:18 PST users could not create workspaces. Root cause: migration left orphan locks. Impacted: workspace-create. Steps taken: cleared locks, added lock TTL. Owners: Taro (migration script), Nina (alert rule).".to_string(),
                ExpectedCase {
                    impacted_keywords: vec!["workspace"],
                    cause_keywords: vec!["migration", "lock"],
                    owners: vec!["Taro", "Nina"],
                },
            ),
        ),
    ];

    let suite = EvalSuite::new("Sentiment Benchmark").with_concurrency(3);

    let report = suite
        .run(cases, move |(text, expected)| {
            let client = client.clone();
            async move {
                let ctx = ContextBuilder::new()
                    .with_system("Extract an incident report. Severity must be one of: Critical, High, Medium, Low. Fill all fields. Use ISO-8601-ish strings for times. Provide at least 2 timeline events and 2 action items. Avoid nulls.")
                    .add_user_text(text);

                let outcome = client
                    .generate_with_metadata::<IncidentReport>(ctx, None, None, None)
                    .await?;

                let passed = validate(&outcome.value, &expected);
                if !passed {
                    tracing::warn!(
                        case = %expected_name(&expected),
                        report = %serde_json::to_string_pretty(&outcome.value).unwrap_or_default(),
                        "Validation failed for case"
                    );
                }

                Ok((outcome, passed))
            }
        })
        .await;

    println!("{}", report);

    Ok(())
}

fn mock_incident() -> serde_json::Value {
    json!({
        "severity": "High",
        "impacted_services": ["checkout", "payments"],
        "root_cause": "Mocked feature flag rollback",
        "start_time": "2024-12-01T09:05Z",
        "end_time": "2024-12-01T09:32Z",
        "timeline": [
            { "time": "2024-12-01T09:05Z", "description": "Errors began" },
            { "time": "2024-12-01T09:32Z", "description": "Mitigation completed" }
        ],
        "actions": [
            { "owner": "Maya", "action": "Rollback feature flag", "due": "2024-12-02" },
            { "owner": "Priya", "action": "Add preflight checks", "due": "2024-12-05" }
        ]
    })
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles
        .iter()
        .any(|needle| haystack.to_lowercase().contains(&needle.to_lowercase()))
}

fn validate(report: &IncidentReport, expected: &ExpectedCase) -> bool {
    let severity_ok = matches!(
        report.severity.to_lowercase().as_str(),
        "critical" | "high" | "medium" | "low"
    );

    let services = report
        .impacted_services
        .iter()
        .map(|s| s.to_lowercase())
        .collect::<Vec<_>>();
    let services_ok = expected
        .impacted_keywords
        .iter()
        .all(|kw| services.iter().any(|s| s.contains(&kw.to_lowercase())));

    let cause_ok = contains_any(&report.root_cause, &expected.cause_keywords);

    let owners_lower = report
        .actions
        .iter()
        .map(|a| a.owner.to_lowercase())
        .collect::<Vec<_>>();
    let owners_ok = expected.owners.iter().all(|o| {
        owners_lower
            .iter()
            .any(|own| own.contains(&o.to_lowercase()))
    });

    let timeline_ok = report.timeline.len() >= 2;
    let actions_ok = report.actions.len() >= 2;
    let times_ok = !report.start_time.is_empty() && !report.end_time.is_empty();

    severity_ok && services_ok && cause_ok && owners_ok && timeline_ok && actions_ok && times_ok
}

fn expected_name(expected: &ExpectedCase) -> String {
    format!(
        "services={:?}, cause={:?}, owners={:?}",
        expected.impacted_keywords, expected.cause_keywords, expected.owners
    )
}

use std::fmt;
use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::{Mutex, Semaphore};

use crate::{models::GenerationOutcome, schema::GeminiStructured, StructuredError};

/// A single evaluation result for a test case.
#[derive(Debug, Clone)]
pub struct EvalResult {
    pub case_name: String,
    pub passed: bool,
    pub score: Option<f64>,
    pub latency: Duration,
    pub prompt_tokens: usize,
    pub response_tokens: usize,
    pub network_attempts: usize,
    pub parse_attempts: usize,
    pub error: Option<String>,
}

impl EvalResult {
    pub fn fail(name: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            case_name: name.into(),
            passed: false,
            score: Some(0.0),
            latency: Duration::default(),
            prompt_tokens: 0,
            response_tokens: 0,
            network_attempts: 0,
            parse_attempts: 0,
            error: Some(error.into()),
        }
    }
}

/// A suite of evaluation cases.
pub struct EvalSuite {
    name: String,
    concurrency: usize,
}

impl EvalSuite {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            concurrency: 5,
        }
    }

    pub fn with_concurrency(mut self, n: usize) -> Self {
        self.concurrency = n.max(1);
        self
    }

    /// Run a list of inputs against an async evaluation function.
    ///
    /// The `evaluator` function receives the input and should return a `Result<(GenerationOutcome<T>, bool)>`
    /// along with a pass/fail flag.
    pub async fn run<I, T, F, Fut>(&self, cases: Vec<(String, I)>, evaluator: F) -> SuiteReport
    where
        I: Send + Sync + 'static,
        T: GeminiStructured + Send + Sync,
        F: Fn(I) -> Fut + Send + Sync + Clone + 'static,
        Fut: Future<Output = Result<(GenerationOutcome<T>, bool), StructuredError>> + Send,
    {
        let results = Arc::new(Mutex::new(Vec::new()));
        let semaphore = Arc::new(Semaphore::new(self.concurrency));
        let mut handles = Vec::new();

        println!(
            "Running suite '{}' with {} cases (concurrency={})...",
            self.name,
            cases.len(),
            self.concurrency
        );

        for (name, input) in cases {
            let eval_fn = evaluator.clone();
            let results = Arc::clone(&results);
            let semaphore = Arc::clone(&semaphore);

            handles.push(tokio::spawn(async move {
                let _permit = semaphore.acquire().await.unwrap();
                let start = Instant::now();

                let eval_res = match eval_fn(input).await {
                    Ok((outcome, passed)) => {
                        let latency = start.elapsed();
                        let usage = outcome.usage.as_ref();
                        EvalResult {
                            case_name: name.clone(),
                            passed,
                            score: Some(if passed { 1.0 } else { 0.0 }),
                            latency,
                            prompt_tokens: usage.and_then(|u| u.prompt_token_count).unwrap_or(0)
                                as usize,
                            response_tokens: usage
                                .and_then(|u| u.candidates_token_count)
                                .unwrap_or(0) as usize,
                            network_attempts: outcome.network_attempts,
                            parse_attempts: outcome.parse_attempts,
                            error: None,
                        }
                    }
                    Err(e) => EvalResult::fail(name.clone(), format!("{e:?}")),
                };

                if eval_res.passed {
                    print!(".");
                } else {
                    print!("F");
                }
                use std::io::Write;
                let _ = std::io::stdout().flush();

                results.lock().await.push(eval_res);
            }));
        }

        for h in handles {
            let _ = h.await;
        }
        println!("\nDone.");

        let final_results = results.lock().await.clone();
        SuiteReport::new(self.name.clone(), final_results)
    }
}

/// Aggregated report of the suite execution.
#[derive(Debug, Clone)]
pub struct SuiteReport {
    pub suite_name: String,
    pub total_cases: usize,
    pub passed: usize,
    pub failed: usize,
    pub avg_latency_ms: u128,
    pub p95_latency_ms: u128,
    pub avg_prompt_tokens: f64,
    pub avg_response_tokens: f64,
    pub total_prompt_tokens: usize,
    pub total_response_tokens: usize,
    pub avg_network_attempts: f64,
    pub avg_parse_attempts: f64,
    pub results: Vec<EvalResult>,
}

impl SuiteReport {
    fn new(name: String, mut results: Vec<EvalResult>) -> Self {
        let total = results.len();
        if total == 0 {
            return Self {
                suite_name: name,
                total_cases: 0,
                passed: 0,
                failed: 0,
                avg_latency_ms: 0,
                p95_latency_ms: 0,
                avg_prompt_tokens: 0.0,
                avg_response_tokens: 0.0,
                total_prompt_tokens: 0,
                total_response_tokens: 0,
                avg_network_attempts: 0.0,
                avg_parse_attempts: 0.0,
                results,
            };
        }

        let passed = results.iter().filter(|r| r.passed).count();
        let failed = total.saturating_sub(passed);

        let total_prompt: usize = results.iter().map(|r| r.prompt_tokens).sum();
        let total_response: usize = results.iter().map(|r| r.response_tokens).sum();
        let total_net: usize = results.iter().map(|r| r.network_attempts).sum();
        let total_parse: usize = results.iter().map(|r| r.parse_attempts).sum();
        let total_latency: u128 = results.iter().map(|r| r.latency.as_millis()).sum();

        results.sort_by(|a, b| a.latency.cmp(&b.latency));
        let p95_idx = ((total as f64 * 0.95).ceil() as usize).saturating_sub(1);
        let p95 = results
            .get(p95_idx)
            .map(|r| r.latency.as_millis())
            .unwrap_or(0);

        Self {
            suite_name: name,
            total_cases: total,
            passed,
            failed,
            avg_latency_ms: total_latency / total as u128,
            p95_latency_ms: p95,
            avg_prompt_tokens: total_prompt as f64 / total as f64,
            avg_response_tokens: total_response as f64 / total as f64,
            total_prompt_tokens: total_prompt,
            total_response_tokens: total_response,
            avg_network_attempts: total_net as f64 / total as f64,
            avg_parse_attempts: total_parse as f64 / total as f64,
            results,
        }
    }
}

impl fmt::Display for SuiteReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n=== Benchmark Report: {} ===", self.suite_name)?;
        writeln!(
            f,
            "Cases: {} | Passed: {} | Failed: {}",
            self.total_cases, self.passed, self.failed
        )?;
        writeln!(
            f,
            "Latency: Avg {:.2}s | P95 {:.2}s",
            self.avg_latency_ms as f64 / 1000.0,
            self.p95_latency_ms as f64 / 1000.0
        )?;
        writeln!(
            f,
            "Tokens (Avg): Prompt {:.0} | Response {:.0}",
            self.avg_prompt_tokens, self.avg_response_tokens
        )?;
        writeln!(
            f,
            "Reliability (Avg): Net Attempts {:.2} | Parse Attempts {:.2}",
            self.avg_network_attempts, self.avg_parse_attempts
        )?;

        if self.failed > 0 {
            writeln!(f, "\n--- Failures ---")?;
            for r in self.results.iter().filter(|r| !r.passed) {
                writeln!(
                    f,
                    "[{}] Error: {} (network_attempts={}, parse_attempts={}, latency_ms={})",
                    r.case_name,
                    r.error.as_deref().unwrap_or("Unknown"),
                    r.network_attempts,
                    r.parse_attempts,
                    r.latency.as_millis()
                )?;
            }
        }
        Ok(())
    }
}

//! Abstract text generation interface.
//!
//! This module defines the `TextGenerator` trait which provides an abstraction
//! over text generation backends. This allows the `RefinementEngine` and other
//! components to work with different LLM providers or testing mocks.

use std::sync::Arc;

use async_trait::async_trait;
use gemini_rust::{Content, Gemini, GenerationConfig, Message, Role};

use crate::error::Result;

/// Abstract interface for text generation.
///
/// This trait decouples text generation logic from specific LLM implementations,
/// enabling easier testing and the potential to use different models or backends
/// for different tasks.
///
/// # Example
///
/// ```rust,ignore
/// use gemini_structured_output::generator::TextGenerator;
///
/// async fn generate_summary(generator: &dyn TextGenerator, document: &str) -> Result<String> {
///     generator.generate_text(
///         Some("You are a summarizer."),
///         &format!("Summarize: {}", document),
///         GenerationConfig::default(),
///     ).await
/// }
/// ```
#[async_trait]
pub trait TextGenerator: Send + Sync {
    /// Generate text based on a prompt and optional system instruction.
    ///
    /// # Arguments
    ///
    /// * `system` - Optional system instruction to guide the model's behavior
    /// * `prompt` - The user prompt to generate a response for
    /// * `config` - Generation configuration (temperature, max tokens, etc.)
    ///
    /// # Returns
    ///
    /// The generated text response, or an error if generation fails.
    async fn generate_text(
        &self,
        system: Option<&str>,
        prompt: &str,
        config: GenerationConfig,
    ) -> Result<String>;
}

/// Implementation of `TextGenerator` for the Gemini client.
///
/// This allows the standard Gemini client to be used anywhere a `TextGenerator`
/// is required.
#[async_trait]
impl TextGenerator for Arc<Gemini> {
    async fn generate_text(
        &self,
        system: Option<&str>,
        prompt: &str,
        config: GenerationConfig,
    ) -> Result<String> {
        let mut builder = self.generate_content();

        if let Some(sys) = system {
            builder = builder.with_system_instruction(sys);
        }

        builder = builder.with_generation_config(config);
        builder = builder.with_message(Message {
            role: Role::User,
            content: Content::text(prompt).with_role(Role::User),
        });

        let response = builder.execute().await?;
        Ok(response.text())
    }
}

/// A wrapper around `Arc<Gemini>` that implements `TextGenerator`.
///
/// This is useful when you need to pass a generator that outlives the borrow
/// of the original client.
#[derive(Clone)]
pub struct GeminiGenerator {
    client: Arc<Gemini>,
}

impl GeminiGenerator {
    /// Create a new generator wrapping a Gemini client.
    pub fn new(client: Arc<Gemini>) -> Self {
        Self { client }
    }
}

#[async_trait]
impl TextGenerator for GeminiGenerator {
    async fn generate_text(
        &self,
        system: Option<&str>,
        prompt: &str,
        config: GenerationConfig,
    ) -> Result<String> {
        self.client.generate_text(system, prompt, config).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    struct MockGenerator {
        response: String,
    }

    #[async_trait]
    impl TextGenerator for MockGenerator {
        async fn generate_text(
            &self,
            _system: Option<&str>,
            _prompt: &str,
            _config: GenerationConfig,
        ) -> Result<String> {
            Ok(self.response.clone())
        }
    }

    #[tokio::test]
    async fn test_mock_generator() {
        let generator = MockGenerator {
            response: "Hello, world!".to_string(),
        };

        let result = generator
            .generate_text(None, "Say hello", GenerationConfig::default())
            .await
            .unwrap();

        assert_eq!(result, "Hello, world!");
    }

    #[tokio::test]
    async fn test_generator_as_trait_object() {
        let generator: Arc<dyn TextGenerator> = Arc::new(MockGenerator {
            response: "Test response".to_string(),
        });

        let result = generator
            .generate_text(Some("System"), "Prompt", GenerationConfig::default())
            .await
            .unwrap();

        assert_eq!(result, "Test response");
    }
}

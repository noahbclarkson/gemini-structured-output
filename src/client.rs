use std::sync::Arc;

use gemini_rust::{
    generation::builder::ContentBuilder, generation::model::UsageMetadata, tools::FunctionCall,
    Gemini, GenerationConfig, Message, Model, Role, SafetySetting, Tool,
};
use serde::{de::DeserializeOwned, Serialize};
use tracing::{debug, info, instrument};

use crate::{
    caching::{CachePolicy, CacheSettings, SchemaCache},
    context::ContextBuilder,
    error::{Result, StructuredError},
    files::FileManager,
    models::{GenerationOutcome, RefinementOutcome},
    patching::{ArrayPatchStrategy, PatchStrategy, RefinementConfig, RefinementEngine},
    schema::{GeminiStructured, StructuredValidator},
    tools::ToolRegistry,
    StructuredRequest,
};

/// Handler used to short-circuit requests during tests.
///
/// The handler receives a lightweight view of the request and must return a JSON string
/// that can be deserialized into the target type.
pub type MockHandler = Arc<dyn Fn(MockRequest) -> Result<String> + Send + Sync>;

/// Minimal view of a structured request passed to [`MockHandler`].
#[derive(Debug, Clone)]
pub struct MockRequest {
    /// The target Rust type name (for logging/debugging only).
    pub target: String,
    /// The system instruction, if any.
    pub system_instruction: Option<String>,
    /// A debug representation of the prompt messages.
    pub prompt_preview: String,
}

/// Strategy for handling model fallbacks during generation and refinement.
///
/// This allows automatic escalation to a more capable model when the primary
/// model fails repeatedly to produce valid output.
#[derive(Clone, Debug, Default)]
pub enum FallbackStrategy {
    /// Use only the primary model (default).
    #[default]
    None,
    /// Escalate to a fallback model after a specified number of failed attempts.
    ///
    /// This is useful when a faster/cheaper model might fail on complex tasks,
    /// allowing automatic escalation to a more capable model.
    Escalate {
        /// Number of failed attempts before switching to the fallback model.
        after_attempts: usize,
        /// The model to escalate to.
        target: Model,
    },
}

/// Options for building a configured content request.
#[derive(Clone)]
pub(crate) struct BuilderOptions<'a> {
    pub tools: &'a [Tool],
    pub config: &'a GenerationConfig,
    pub cache_settings: &'a Option<CacheSettings>,
    pub system_instruction: &'a Option<String>,
    pub safety_settings: &'a Option<Vec<SafetySetting>>,
}

/// Global configuration options for the client.
#[derive(Clone, Debug)]
pub struct ClientConfig {
    /// Default temperature for generation (default: 0.1)
    pub default_temperature: f32,
    /// Default max retries for transient errors (default: 3)
    pub default_retries: usize,
    /// Default max parse attempts (default: 3)
    pub default_parse_attempts: usize,
    /// Default max tool steps (default: 5)
    pub default_tool_steps: usize,
    /// Array patching strategy for refinement (default: ReplaceWhole)
    pub array_strategy: ArrayPatchStrategy,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            default_temperature: 0.1,
            default_retries: 3,
            default_parse_attempts: 3,
            default_tool_steps: 5,
            array_strategy: ArrayPatchStrategy::ReplaceWhole,
        }
    }
}

/// Builder for [`StructuredClient`].
pub struct StructuredClientBuilder {
    api_key: String,
    model: Model,
    cache_policy: CachePolicy,
    refinement_retries: usize,
    refinement_temperature: f32,
    refinement_network_retries: usize,
    refinement_strategy: PatchStrategy,
    fallback_strategy: FallbackStrategy,
    config: ClientConfig,
    mock_handler: Option<MockHandler>,
}

impl StructuredClientBuilder {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: Model::Gemini25Flash,
            cache_policy: CachePolicy::Disabled,
            refinement_retries: 3,
            refinement_temperature: 0.0,
            refinement_network_retries: 3,
            refinement_strategy: PatchStrategy::PartialApply,
            fallback_strategy: FallbackStrategy::default(),
            config: ClientConfig::default(),
            mock_handler: None,
        }
    }

    /// Set the model to use.
    pub fn with_model(mut self, model: Model) -> Self {
        self.model = model;
        self
    }

    /// Enable caching with the specified policy.
    pub fn with_cache_policy(mut self, policy: CachePolicy) -> Self {
        self.cache_policy = policy;
        self
    }

    /// Set maximum refinement retry attempts.
    pub fn with_refinement_retries(mut self, retries: usize) -> Self {
        self.refinement_retries = retries.max(1);
        self
    }

    /// Set temperature for refinement operations.
    pub fn with_refinement_temperature(mut self, temperature: f32) -> Self {
        self.refinement_temperature = temperature;
        self
    }

    /// Coerce `null` to `0` for numeric fields during refinement validation (default: true).
    /// Number of network retries for transient errors (e.g., 429/503) during refinement.
    pub fn with_refinement_network_retries(mut self, retries: usize) -> Self {
        self.refinement_network_retries = retries;
        self
    }

    /// Strategy for applying patches during refinement (atomic vs partial).
    pub fn with_refinement_strategy(mut self, strategy: PatchStrategy) -> Self {
        self.refinement_strategy = strategy;
        self
    }

    /// Set the fallback strategy for model escalation.
    ///
    /// When enabled, the client will automatically switch to a more capable model
    /// if the primary model fails repeatedly to produce valid output.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use gemini_structured_output::prelude::*;
    ///
    /// let client = StructuredClientBuilder::new("api-key")
    ///     .with_model(Model::Gemini25Flash)
    ///     .with_fallback_strategy(FallbackStrategy::Escalate {
    ///         after_attempts: 2,
    ///         target: Model::Gemini25Pro,
    ///     })
    ///     .build()?;
    /// ```
    pub fn with_fallback_strategy(mut self, strategy: FallbackStrategy) -> Self {
        self.fallback_strategy = strategy;
        self
    }

    /// Set the default generation temperature.
    pub fn with_default_temperature(mut self, temperature: f32) -> Self {
        self.config.default_temperature = temperature;
        self
    }

    /// Set the default number of retries for transient errors.
    pub fn with_default_retries(mut self, retries: usize) -> Self {
        self.config.default_retries = retries;
        self
    }

    /// Set the default maximum parse attempts.
    pub fn with_default_parse_attempts(mut self, attempts: usize) -> Self {
        self.config.default_parse_attempts = attempts;
        self
    }

    /// Set the default maximum tool steps.
    pub fn with_default_tool_steps(mut self, steps: usize) -> Self {
        self.config.default_tool_steps = steps;
        self
    }

    /// Set the array patching strategy for refinement.
    pub fn with_array_strategy(mut self, strategy: ArrayPatchStrategy) -> Self {
        self.config.array_strategy = strategy.clone();
        self
    }

    /// Apply a complete client configuration.
    pub fn with_config(mut self, config: ClientConfig) -> Self {
        self.config = config;
        self
    }

    /// Provide a mock handler to intercept all requests.
    ///
    /// This is primarily intended for unit tests where network calls should be avoided.
    pub fn with_mock(
        mut self,
        handler: impl Fn(MockRequest) -> Result<String> + Send + Sync + 'static,
    ) -> Self {
        self.mock_handler = Some(Arc::new(handler));
        self
    }

    /// Build the client.
    pub fn build(self) -> Result<StructuredClient> {
        let client = Arc::new(Gemini::with_model(&self.api_key, self.model.clone())?);

        // Create fallback client if escalation is enabled
        let fallback_client = match &self.fallback_strategy {
            FallbackStrategy::Escalate { target, .. } => {
                Some(Arc::new(Gemini::with_model(&self.api_key, target.clone())?))
            }
            FallbackStrategy::None => None,
        };

        let refiner_config = RefinementConfig {
            max_retries: self.refinement_retries,
            temperature: self.refinement_temperature,
            patch_strategy: self.refinement_strategy.clone(),
            array_strategy: self.config.array_strategy.clone(),
            network_retries: self.refinement_network_retries,
            fallback_strategy: self.fallback_strategy.clone(),
        };

        let refiner = RefinementEngine::new(client.clone(), fallback_client.clone())
            .with_config(refiner_config);

        Ok(StructuredClient {
            client: client.clone(),
            fallback_client,
            fallback_strategy: self.fallback_strategy,
            model: self.model,
            file_manager: FileManager::new(client.clone()),
            refiner,
            cache: SchemaCache::new(client.clone(), self.cache_policy),
            config: self.config,
            mock_handler: self.mock_handler,
        })
    }
}

#[derive(Clone)]
pub struct StructuredClient {
    pub client: Arc<Gemini>,
    pub fallback_client: Option<Arc<Gemini>>,
    pub fallback_strategy: FallbackStrategy,
    pub model: Model,
    pub file_manager: FileManager,
    refiner: RefinementEngine,
    cache: SchemaCache,
    config: ClientConfig,
    pub(crate) mock_handler: Option<MockHandler>,
}

impl StructuredClient {
    /// Quick generation with minimal configuration.
    ///
    /// This is a convenience method for simple use cases. For more control,
    /// use the `request()` builder.
    ///
    /// # Example
    /// ```rust,no_run
    /// # use gemini_structured_output::StructuredClientBuilder;
    /// # use schemars::JsonSchema;
    /// # use serde::{Deserialize, Serialize};
    /// #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
    /// struct Person { name: String }
    ///
    /// # async fn example() -> std::result::Result<(), Box<dyn std::error::Error>> {
    /// let client = StructuredClientBuilder::new("key").build()?;
    /// let person: Person = client.quick_generate("Name: Alice").await?;
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(skip_all, fields(target = std::any::type_name::<T>()))]
    pub async fn quick_generate<T>(&self, prompt: impl Into<String>) -> Result<T>
    where
        T: GeminiStructured
            + StructuredValidator
            + Serialize
            + DeserializeOwned
            + Clone
            + Send
            + Sync
            + 'static,
    {
        let result = self
            .request::<T>()
            .user_text(prompt)
            .temperature(self.config.default_temperature)
            .execute()
            .await?;
        Ok(result.value)
    }

    /// Quick generation with a system instruction.
    #[instrument(skip_all, fields(target = std::any::type_name::<T>()))]
    pub async fn quick_generate_with_system<T>(
        &self,
        system: impl Into<String>,
        prompt: impl Into<String>,
    ) -> Result<T>
    where
        T: GeminiStructured
            + StructuredValidator
            + Serialize
            + DeserializeOwned
            + Clone
            + Send
            + Sync
            + 'static,
    {
        let result = self
            .request::<T>()
            .system(system)
            .user_text(prompt)
            .temperature(self.config.default_temperature)
            .execute()
            .await?;
        Ok(result.value)
    }

    /// Generate a structured response validated by `T`'s schema.
    #[instrument(skip_all, fields(target = std::any::type_name::<T>()))]
    pub async fn generate<T>(&self, ctx: ContextBuilder, tools: Option<ToolRegistry>) -> Result<T>
    where
        T: GeminiStructured + DeserializeOwned,
    {
        Ok(self
            .generate_with_metadata::<T>(ctx, tools, None, None)
            .await?
            .value)
    }

    /// Same as [`generate`] but returns parsed value plus metadata.
    pub async fn generate_with_metadata<T>(
        &self,
        ctx: ContextBuilder,
        tools: Option<ToolRegistry>,
        generation_config: Option<GenerationConfig>,
        cache_settings: Option<CacheSettings>,
    ) -> Result<GenerationOutcome<T>>
    where
        T: GeminiStructured + DeserializeOwned,
    {
        let (system_instruction, contents) = ctx.build();
        let tools_vec: Vec<Tool> = tools.as_ref().map(|t| t.definitions()).unwrap_or_default();
        let mut messages = Vec::new();
        for content in contents {
            let role = content.role.clone().unwrap_or(Role::User);
            messages.push(Message {
                role: role.clone(),
                content: content.clone().with_role(role),
            });
        }

        self.execute_request::<T>(
            messages,
            system_instruction,
            tools_vec,
            generation_config.unwrap_or_default(),
            cache_settings,
        )
        .await
    }

    /// Refine an existing value using a JSON Patch feedback loop.
    #[instrument(skip_all, fields(target = std::any::type_name::<T>()))]
    pub async fn refine<T>(&self, current: &T, instruction: &str) -> Result<RefinementOutcome<T>>
    where
        T: GeminiStructured + StructuredValidator + Serialize + DeserializeOwned + Clone,
    {
        self.refiner.refine(current, instruction).await
    }

    /// Access the underlying Gemini client when low-level controls are required.
    pub fn raw(&self) -> Arc<Gemini> {
        self.client.clone()
    }

    /// Get the current client configuration.
    pub fn config(&self) -> &ClientConfig {
        &self.config
    }

    /// Get the fallback strategy.
    pub fn fallback_strategy(&self) -> &FallbackStrategy {
        &self.fallback_strategy
    }

    /// Select the appropriate client based on the fallback strategy and attempt count.
    ///
    /// Returns a tuple of (client, escalated) where `escalated` is true if this is
    /// using the fallback model.
    pub(crate) fn select_client(&self, attempt: usize) -> (&Arc<Gemini>, bool) {
        match &self.fallback_strategy {
            FallbackStrategy::Escalate {
                after_attempts,
                target: _,
            } if attempt > *after_attempts && self.fallback_client.is_some() => {
                (self.fallback_client.as_ref().unwrap(), true)
            }
            _ => (&self.client, false),
        }
    }

    /// Start building a fluent structured request.
    pub fn request<T>(&self) -> StructuredRequest<'_, T>
    where
        T: GeminiStructured
            + StructuredValidator
            + Serialize
            + DeserializeOwned
            + Clone
            + Send
            + Sync
            + 'static,
    {
        StructuredRequest::new(self)
            .max_parse_attempts(self.config.default_parse_attempts)
            .max_tool_steps(self.config.default_tool_steps)
            .retries(self.config.default_retries)
            .temperature(self.config.default_temperature)
    }

    /// Generate structured data using a runtime-provided JSON Schema.
    ///
    /// This is useful when the response shape is only known at runtime (e.g., derived
    /// from user-uploaded content or dynamically constructed prompts).
    pub async fn generate_dynamic(
        &self,
        json_schema: serde_json::Value,
        ctx: ContextBuilder,
        generation_config: Option<GenerationConfig>,
    ) -> Result<serde_json::Value> {
        let (system_instruction, contents) = ctx.build();
        let mut messages = Vec::new();
        for content in contents {
            let role = content.role.clone().unwrap_or(Role::User);
            messages.push(Message {
                role: role.clone(),
                content: content.clone().with_role(role),
            });
        }

        let mut generation_config = generation_config.unwrap_or_default();
        generation_config.response_schema = Some(json_schema);
        generation_config
            .response_mime_type
            .get_or_insert_with(|| "application/json".to_string());
        generation_config
            .temperature
            .get_or_insert(self.config.default_temperature);

        let mut builder = self.client.generate_content();
        for msg in messages {
            builder = builder.with_message(msg);
        }
        if let Some(system) = system_instruction {
            builder = builder.with_system_instruction(system);
        }

        let response = builder
            .with_generation_config(generation_config)
            .execute()
            .await?;
        let text = response.text();
        let cleaned = crate::request::clean_json_text(&text);
        serde_json::from_str::<serde_json::Value>(&cleaned)
            .map_err(|e| StructuredError::parse_error(e, &cleaned))
    }

    pub(crate) async fn execute_request<T>(
        &self,
        contents: Vec<Message>,
        system_instruction: Option<String>,
        tools: Vec<Tool>,
        config: GenerationConfig,
        cache_settings: Option<CacheSettings>,
    ) -> Result<GenerationOutcome<T>>
    where
        T: GeminiStructured + DeserializeOwned,
    {
        if let Some(mock) = &self.mock_handler {
            let preview = contents
                .iter()
                .map(|m| format!("{m:?}"))
                .collect::<Vec<_>>()
                .join("\n---\n");
            let request = MockRequest {
                target: std::any::type_name::<T>().to_string(),
                system_instruction: system_instruction.clone(),
                prompt_preview: preview,
            };
            let raw = (mock)(request)?;
            let parsed: T =
                serde_json::from_str(&raw).map_err(|e| StructuredError::parse_error(e, &raw))?;
            return Ok(GenerationOutcome::new(
                parsed,
                None,
                vec![],
                None,
                None,
                0,
                0,
            ));
        }

        let builder = self
            .configured_builder::<T>(
                &contents,
                BuilderOptions {
                    tools: &tools,
                    config: &config,
                    cache_settings: &cache_settings,
                    system_instruction: &system_instruction,
                    safety_settings: &None,
                },
            )
            .await?;

        let response = builder.execute().await?;
        let text = response.text();
        let parsed: T = serde_json::from_str(&text)?;

        let usage: Option<UsageMetadata> = response.usage_metadata.clone();
        let function_calls: Vec<FunctionCall> =
            response.function_calls().into_iter().cloned().collect();

        Ok(GenerationOutcome::new(
            parsed,
            usage,
            function_calls,
            response.model_version.clone(),
            response.response_id.clone(),
            0,
            1,
        ))
    }

    pub(crate) async fn configured_builder<T>(
        &self,
        messages: &[Message],
        opts: BuilderOptions<'_>,
    ) -> Result<ContentBuilder>
    where
        T: GeminiStructured,
    {
        self.configured_builder_with_client::<T>(&self.client, messages, opts)
            .await
    }

    /// Create a configured builder using a specific client.
    ///
    /// This allows using either the primary or fallback client for generation.
    pub(crate) async fn configured_builder_with_client<T>(
        &self,
        client: &Arc<Gemini>,
        messages: &[Message],
        opts: BuilderOptions<'_>,
    ) -> Result<ContentBuilder>
    where
        T: GeminiStructured,
    {
        let BuilderOptions {
            tools,
            config,
            cache_settings,
            system_instruction,
            safety_settings,
        } = opts;
        let schema = T::gemini_schema();
        let mut config = config.clone();
        let has_tools = !tools.is_empty();
        let model_str = self.model.as_str();
        let is_gemini_3 = model_str.contains("gemini-3") || model_str.contains("gemini-experiment");

        let mut final_system_instruction = system_instruction.clone();

        // Log the schema that will be enforced for this request and how it is applied.
        let schema_json = serde_json::to_string_pretty(&schema)
            .unwrap_or_else(|_| "Unable to serialize schema".to_string());

        if has_tools {
            if is_gemini_3 {
                // Gemini 3: enable strict JSON outputs alongside tools.
                debug!("Gemini 3 detected: enforcing JSON schema with tools enabled");
                info!("Applying response schema via generation config (tools enabled):\n{schema_json}");
                config.response_schema = Some(schema);
                config
                    .response_mime_type
                    .get_or_insert_with(|| "application/json".to_string());
            } else {
                // Legacy models: inject schema into system prompt instead of forcing mime/schema in config.
                debug!("Legacy model with tools: injecting schema into system prompt");
                info!("Embedding schema into system prompt (tools enabled legacy path):\n{schema_json}");
                config.response_schema = None;
                config.response_mime_type = None;

                let schema_instruction = format!(
                    "You must output valid JSON matching this schema exactly:\n{}",
                    serde_json::to_string_pretty(&schema).unwrap_or_default()
                );

                final_system_instruction = Some(match final_system_instruction {
                    Some(existing) => format!("{}\n\n{}", existing, schema_instruction),
                    None => schema_instruction,
                });
            }
        } else {
            info!("Applying response schema via generation config (no tools):\n{schema_json}");
            config.response_schema = Some(schema);
            config
                .response_mime_type
                .get_or_insert_with(|| "application/json".to_string());
        }

        config
            .temperature
            .get_or_insert(self.config.default_temperature);

        let mut builder = client.generate_content();
        for msg in messages {
            builder = builder.with_message(msg.clone());
        }

        if let Some(system) = final_system_instruction {
            let cache_key = cache_settings
                .as_ref()
                .and_then(|c| c.key.clone())
                .unwrap_or_else(|| SchemaCache::cache_key::<T>(&system, tools));
            let ttl_override = cache_settings.as_ref().and_then(|c| c.ttl_override);

            if let Some(handle) = self
                .cache
                .get_or_create(&cache_key, &system, tools, ttl_override)
                .await?
            {
                builder = builder.with_cached_content(&handle);
            } else {
                builder = builder.with_system_instruction(system.clone());
            }
        }

        for tool in tools {
            builder = builder.with_tool(tool.clone());
        }

        if let Some(safety) = safety_settings {
            builder = builder.with_safety_settings(safety.clone());
        }

        Ok(builder.with_generation_config(config))
    }
}

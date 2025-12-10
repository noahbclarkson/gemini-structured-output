use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use gemini_rust::{FunctionDeclaration, Tool};
use schemars::JsonSchema;
use serde::{de::DeserializeOwned, Serialize};
use serde_json::Value;

use crate::error::{Result, StructuredError};

/// A dynamic error type for tool execution.
pub type ToolError = Box<dyn std::error::Error + Send + Sync>;

/// A handler that takes a JSON argument and returns a JSON result (async).
type HandlerFn = dyn Fn(
        Value,
    ) -> Pin<Box<dyn Future<Output = std::result::Result<Value, ToolError>> + Send + 'static>>
    + Send
    + Sync;

#[derive(Clone, Default)]
pub struct ToolRegistry {
    tools: Vec<Tool>,
    handlers: Arc<HashMap<String, Arc<HandlerFn>>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: Vec::new(),
            handlers: Arc::new(HashMap::new()),
        }
    }

    /// Register a function tool using typed arguments and response payloads (no handler).
    pub fn register<Args, Resp>(mut self, name: &str, description: &str) -> Self
    where
        Args: JsonSchema + Serialize,
        Resp: JsonSchema + Serialize,
    {
        let declaration = FunctionDeclaration::new(name, description, None)
            .with_parameters::<Args>()
            .with_response::<Resp>();

        self.tools.push(Tool::new(declaration));
        self
    }

    /// Register a function tool with an async handler implementation.
    pub fn register_with_handler<Args, Resp, F, Fut>(
        mut self,
        name: &str,
        description: &str,
        handler: F,
    ) -> Self
    where
        Args: JsonSchema + Serialize + DeserializeOwned + Send + Sync + 'static,
        Resp: JsonSchema + Serialize + Send + Sync + 'static,
        F: Fn(Args) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = std::result::Result<Resp, ToolError>> + Send + 'static,
    {
        let declaration = FunctionDeclaration::new(name, description, None)
            .with_parameters::<Args>()
            .with_response::<Resp>();
        self.tools.push(Tool::new(declaration));

        let name_owned = name.to_string();
        let handler_arc: Arc<F> = Arc::new(handler);
        let handler_ref = handler_arc.clone();
        let wrapper: Arc<HandlerFn> = Arc::new(move |args_val: Value| {
            let handler_call = handler_ref.clone();
            let fut = async move {
                let args: Args = serde_json::from_value(args_val)
                    .map_err(|e| Box::new(StructuredError::Json(e)) as ToolError)?;
                let result = handler_call(args).await?;
                let res_val = serde_json::to_value(result)
                    .map_err(|e| Box::new(StructuredError::Json(e)) as ToolError)?;
                Ok(res_val)
            };
            Box::pin(fut)
        });

        let mut new_handlers = (*self.handlers).clone();
        new_handlers.insert(name_owned, wrapper);
        self.handlers = Arc::new(new_handlers);

        self
    }

    /// Add an existing tool instance (e.g., Google Search or Code Execution).
    pub fn with_tool(mut self, tool: Tool) -> Self {
        self.tools.push(tool);
        self
    }

    /// Convenience for adding the Google Search tool.
    pub fn with_google_search(self) -> Self {
        self.with_tool(Tool::google_search())
    }

    /// Convenience for enabling the Code Execution tool.
    pub fn with_code_execution(self) -> Self {
        self.with_tool(Tool::code_execution())
    }

    pub fn definitions(&self) -> Vec<Tool> {
        self.tools.clone()
    }

    pub async fn execute(&self, name: &str, args: Value) -> Result<Value> {
        if let Some(handler) = self.handlers.get(name) {
            handler(args)
                .await
                .map_err(|e| StructuredError::Context(e.to_string()))
        } else {
            Err(StructuredError::Context(format!(
                "No handler registered for tool: {name}"
            )))
        }
    }

    /// Register a tool using a registrar function.
    ///
    /// This is designed to work with the `#[gemini_tool]` macro which generates
    /// registrar functions for tools.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use gemini_structured_output::prelude::*;
    ///
    /// #[gemini_tool(description = "Look up stock price")]
    /// async fn get_stock_price(args: StockRequest) -> Result<StockPrice, ToolError> {
    ///     // ...
    /// }
    ///
    /// let registry = ToolRegistry::new()
    ///     .register_tool(get_stock_price_tool::registrar());
    /// ```
    pub fn register_tool<F>(self, registrar: F) -> Self
    where
        F: FnOnce(ToolRegistry) -> ToolRegistry,
    {
        registrar(self)
    }
}

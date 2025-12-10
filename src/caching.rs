use std::{collections::HashMap, sync::Arc, time::Duration};

use gemini_rust::{
    cache::{CachedContentHandle, Error as CacheError},
    ClientError, Gemini, Tool,
};
use sha2::{Digest, Sha256};
use tokio::sync::Mutex;
use tracing::{debug, warn};

use crate::{error::Result, schema::GeminiStructured};

#[derive(Clone, Copy)]
pub enum CachePolicy {
    Disabled,
    Enabled { ttl: Duration },
}

/// Per-call cache overrides for caching behavior.
#[derive(Clone, Default)]
pub struct CacheSettings {
    pub key: Option<String>,
    pub ttl_override: Option<Duration>,
}

impl CacheSettings {
    pub fn with_key(key: impl Into<String>) -> Self {
        Self {
            key: Some(key.into()),
            ..Default::default()
        }
    }

    pub fn with_ttl(ttl: Duration) -> Self {
        Self {
            ttl_override: Some(ttl),
            ..Default::default()
        }
    }

    pub fn with_key_and_ttl(key: impl Into<String>, ttl: Duration) -> Self {
        Self {
            key: Some(key.into()),
            ttl_override: Some(ttl),
        }
    }
}

/// Lightweight cache helper to avoid re-uploading heavy schemas or prompts.
#[derive(Clone)]
pub struct SchemaCache {
    client: Arc<Gemini>,
    inner: Arc<Mutex<HashMap<String, CachedContentHandle>>>,
    policy: CachePolicy,
}

impl SchemaCache {
    pub fn new(client: Arc<Gemini>, policy: CachePolicy) -> Self {
        Self {
            client,
            inner: Arc::new(Mutex::new(HashMap::new())),
            policy,
        }
    }

    pub fn policy(&self) -> CachePolicy {
        self.policy
    }

    /// Builds a deterministic cache key from system text, schema, and tool set.
    pub fn cache_key<T: GeminiStructured>(system: &str, tools: &[Tool]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(system.as_bytes());
        hasher.update(T::gemini_schema_hash().as_bytes());
        for tool in tools {
            hasher.update(format!("{tool:?}").as_bytes());
        }
        let digest = hasher.finalize();
        let suffix = digest
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect::<String>();
        format!("gso-cache-{suffix}")
    }

    /// Create or reuse a cached content handle. Returns `None` when caching is disabled.
    pub async fn get_or_create(
        &self,
        name: &str,
        system_instruction: &str,
        tools: &[Tool],
        ttl_override: Option<Duration>,
    ) -> Result<Option<CachedContentHandle>> {
        match self.policy {
            CachePolicy::Disabled => Ok(None),
            CachePolicy::Enabled { ttl } => {
                // Heuristic: skip caching when content is likely too small to meet API limits.
                // ~4 chars ≈ 1 token. Use 8000 chars (~2000 tokens) as a conservative cutoff.
                let estimated_chars = system_instruction.len() + tools.len() * 100;
                if estimated_chars < 8000 {
                    debug!(
                        cache_key = name,
                        estimated_chars,
                        "Skipping cache creation because content is likely below minimum size"
                    );
                    return Ok(None);
                }

                let ttl = ttl_override.unwrap_or(ttl);
                // Fast path: local map
                if let Some(existing) = self.inner.lock().await.get(name).cloned() {
                    return Ok(Some(existing));
                }

                let mut builder = self
                    .client
                    .create_cache()
                    .with_display_name(name.to_string())?
                    .with_system_instruction(system_instruction.to_string())
                    .with_ttl(ttl);

                if !tools.is_empty() {
                    builder = builder.with_tools(tools.to_vec());
                }

                match builder.execute().await {
                    Ok(handle) => {
                        self.inner
                            .lock()
                            .await
                            .insert(name.to_string(), handle.clone());
                        Ok(Some(handle))
                    }
                    Err(CacheError::Client { source }) => {
                        if let ClientError::BadResponse {
                            code: 400,
                            description: Some(desc),
                        } = source.as_ref()
                        {
                            if desc.contains("Cached content is too small") {
                                // Fall back to uncached flow for tiny prompts/schemas.
                                warn!(
                                    "Cache creation failed because content is too small; continuing without cache"
                                );
                                return Ok(None);
                            }
                        }
                        Err(CacheError::Client { source }.into())
                    }
                    Err(e) => Err(e.into()),
                }
            }
        }
    }
}

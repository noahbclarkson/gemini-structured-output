use std::collections::HashMap;

use chrono::{DateTime, Utc};
use gemini_rust::{Content, Message, Role};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::{
    context::ContextBuilder,
    error::{Result, StructuredError},
    models::RefinementOutcome,
    schema::{GeminiStructured, StructuredValidator},
    StructuredClient,
};

/// Classification for session history entries to separate conversation from state changes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EntryKind {
    Conversation,
    StateChange {
        patch_summary: String,
        effect_summary: Option<String>,
    },
    SystemNote,
}

/// Rich history entry that carries metadata for persistence and UI rendering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionEntry {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub kind: EntryKind,
    pub message: Message,
    pub metadata: HashMap<String, String>,
}

impl SessionEntry {
    pub fn new_chat(role: Role, text: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            kind: EntryKind::Conversation,
            message: Message {
                role: role.clone(),
                content: Content::text(text).with_role(role),
            },
            metadata: HashMap::new(),
        }
    }

    pub fn new_state_change(
        patch_summary: impl Into<String>,
        effect_summary: Option<String>,
        message_role: Role,
        message_text: impl Into<String>,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            kind: EntryKind::StateChange {
                patch_summary: patch_summary.into(),
                effect_summary,
            },
            message: Message {
                role: message_role.clone(),
                content: Content::text(message_text).with_role(message_role),
            },
            metadata: HashMap::new(),
        }
    }

    pub fn new_system_note(text: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            kind: EntryKind::SystemNote,
            message: Message {
                role: Role::User,
                content: Content::text(text).with_role(Role::User),
            },
            metadata: HashMap::new(),
        }
    }

    pub fn with_meta(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

/// Describes the observed impact of a change to help models reason about downstream effects.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeEffect {
    pub description: String,
    pub is_positive: Option<bool>,
}

/// Represents a pending AI-proposed change that awaits user approval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingChange<C> {
    pub proposed_config: C,
    pub patch: json_patch::Patch,
    pub reasoning: Option<String>,
}

/// Top-level container for managing stateful, human-in-the-loop interactions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractiveSession<C, O> {
    /// The currently accepted configuration.
    pub config: C,
    /// Derived output generated from the configuration (e.g., a forecast).
    pub output: Option<O>,
    /// Conversation and state change history with metadata.
    pub history: Vec<SessionEntry>,
    /// AI-proposed change awaiting review.
    pub pending_change: Option<PendingChange<C>>,
}

impl<C, O> InteractiveSession<C, O>
where
    C: GeminiStructured
        + StructuredValidator
        + Serialize
        + DeserializeOwned
        + Clone
        + Send
        + Sync
        + 'static,
    O: Serialize + DeserializeOwned + Clone + Send + Sync + 'static,
{
    pub fn new(initial_config: C, initial_output: Option<O>) -> Self {
        Self {
            config: initial_config,
            output: initial_output,
            history: Vec::new(),
            pending_change: None,
        }
    }

    /// Replace the derived output after recomputing it externally.
    pub fn update_output(&mut self, output: Option<O>) {
        self.output = output;
    }

    /// Build an anchored system prompt plus message history for the next turn.
    fn build_context(&self, _user_query: &str) -> Result<(String, Vec<Message>)> {
        let mut system_prompt = format!(
            "ROLE: You are an assistant managing a configuration workflow.\n\
             \n\
             === CURRENT CONFIGURATION (TRUTH) ===\n\
             {}\n\
             \n\
             === DERIVED OUTPUT ===\n\
             {}\n\
             \n\
             INSTRUCTIONS:\n\
             - Treat the configuration above as the source of truth; older values in history may be stale.\n\
             - Use history for rationale and prior discussion, but resolve conflicts in favor of the configuration block.\n",
            serde_json::to_string_pretty(&self.config)?,
            serde_json::to_string_pretty(&self.output)?
        );

        if let Some(pending) = &self.pending_change {
            system_prompt.push_str("\nPENDING CHANGE:\n");
            system_prompt.push_str(&serde_json::to_string_pretty(&pending.patch)?);
        }

        let messages: Vec<Message> = self
            .history
            .iter()
            .map(|entry| entry.message.clone())
            .collect();

        Ok((system_prompt, messages))
    }

    /// Ask a free-form question about the current state while keeping the config as system context.
    pub async fn chat(
        &mut self,
        client: &StructuredClient,
        user_query: impl Into<String>,
    ) -> Result<String> {
        let user_query = user_query.into();
        let (system_prompt, history_messages) = self.build_context(&user_query)?;

        let ctx = ContextBuilder::new()
            .with_system(system_prompt)
            .add_history(history_messages)
            .add_user_text(&user_query);

        let response_text: String = client.generate(ctx, None).await?;

        self.history
            .push(SessionEntry::new_chat(Role::User, user_query));
        self.history
            .push(SessionEntry::new_chat(Role::Model, response_text.clone()));

        Ok(response_text)
    }

    /// Ask the AI to propose a configuration change and stage it for review.
    pub async fn request_change(
        &mut self,
        client: &StructuredClient,
        instruction: impl Into<String>,
    ) -> Result<&PendingChange<C>> {
        let instruction = instruction.into();
        let outcome = client
            .refine(self.config.clone(), instruction.clone())
            .execute()
            .await?;

        let proposed_config = outcome.value;
        let patch = if let Some(p) = outcome.patch {
            p
        } else {
            let old_json = serde_json::to_value(&self.config)?;
            let new_json = serde_json::to_value(&proposed_config)?;
            json_patch::diff(&old_json, &new_json)
        };

        self.pending_change = Some(PendingChange {
            proposed_config,
            patch,
            reasoning: Some(instruction.clone()),
        });

        let pending = self.pending_change.as_ref().unwrap();
        let patch_text = serde_json::to_string_pretty(&pending.patch)?;

        self.history
            .push(SessionEntry::new_chat(Role::User, instruction));
        self.history.push(
            SessionEntry::new_state_change(
                "Proposed change awaiting approval",
                None,
                Role::Model,
                format!("Proposed change ready for review:\n{}", patch_text),
            )
            .with_meta("type", "ai_proposal"),
        );

        Ok(pending)
    }

    /// Accept the staged change and promote it to the active configuration.
    pub fn accept_change(&mut self) -> Result<&C> {
        let pending = self
            .pending_change
            .take()
            .ok_or_else(|| StructuredError::Context("No pending change to accept".to_string()))?;

        self.config = pending.proposed_config;
        self.history
            .push(SessionEntry::new_system_note("Change accepted."));
        Ok(&self.config)
    }

    /// Decline the staged change.
    pub fn decline_change(&mut self) -> Result<()> {
        if self.pending_change.is_some() {
            self.pending_change = None;
            self.history
                .push(SessionEntry::new_system_note("Change declined."));
            Ok(())
        } else {
            Err(StructuredError::Context(
                "No pending change to decline".to_string(),
            ))
        }
    }

    /// Apply a user-made configuration change, update output, and record semantic effects.
    pub fn apply_manual_change(
        &mut self,
        new_config: C,
        new_output: O,
        effect: Option<ChangeEffect>,
    ) -> Result<json_patch::Patch> {
        let old_json = serde_json::to_value(&self.config)?;
        let new_json = serde_json::to_value(&new_config)?;
        let patch = json_patch::diff(&old_json, &new_json);

        let output_patch = if let Some(old_output) = &self.output {
            let old_output_json = serde_json::to_value(old_output)?;
            let new_output_json = serde_json::to_value(&new_output)?;
            let diff = json_patch::diff(&old_output_json, &new_output_json);
            Some(diff)
        } else {
            None
        };

        self.config = new_config;
        self.output = Some(new_output);
        self.pending_change = None;

        let mut text = format!(
            "SYSTEM UPDATE: The user manually modified the configuration.\nTechnical Changes: {}\n",
            serde_json::to_string(&patch)?
        );

        if let Some(diff) = &output_patch {
            if !diff.0.is_empty() {
                text.push_str(&format!(
                    "Automatic Output Delta: {}\n",
                    serde_json::to_string(diff)?
                ));
            }
        }

        let effect_summary = if let Some(eff) = effect {
            text.push_str(&format!("Observed Effect on Output: {}\n", eff.description));
            Some(eff.description)
        } else {
            None
        };

        let entry = SessionEntry::new_state_change(
            "Manual configuration update",
            effect_summary,
            Role::User,
            text,
        )
        .with_meta("type", "manual_override");

        self.history.push(entry);

        Ok(patch)
    }

    /// Squash a refinement outcome into a single history entry.
    pub fn record_refinement_outcome(
        &mut self,
        instruction: String,
        outcome: &RefinementOutcome<C>,
    ) {
        let attempts = outcome.attempts.len();
        if let Some(final_patch) = &outcome.patch {
            let summary = format!(
                "Applied changes based on: '{}'. (Success after {} attempts)",
                instruction, attempts
            );
            let patch_json =
                serde_json::to_string_pretty(final_patch).unwrap_or_else(|_| "[]".to_string());

            self.history.push(
                SessionEntry::new_state_change(
                    summary,
                    None,
                    Role::Model,
                    format!(
                        "I have updated the configuration.\n\nChanges:\n```json\n{}\n```",
                        patch_json
                    ),
                )
                .with_meta("attempts", &attempts.to_string()),
            );
            self.config = outcome.value.clone();
        } else {
            let last_error = outcome
                .attempts
                .last()
                .and_then(|a| a.error.as_ref())
                .map(|e| e.as_str())
                .unwrap_or("Unknown error");

            self.history.push(SessionEntry::new_system_note(format!(
                "Failed to apply change: '{}'. Gave up after {} attempts.\nLast Error: {}",
                instruction, attempts, last_error
            )));
        }
    }
}

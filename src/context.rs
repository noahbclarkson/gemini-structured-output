use std::sync::Arc;

use gemini_rust::{Content, FileHandle, Message, Part, Role};

use crate::{error::Result, files::FileManager};

/// Builder that assembles system instructions and conversation history.
#[derive(Clone, Default)]
pub struct ContextBuilder {
    system_instruction: Option<String>,
    messages: Vec<Message>,
}

impl ContextBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_system(mut self, instruction: impl Into<String>) -> Self {
        self.system_instruction = Some(instruction.into());
        self
    }

    pub fn add_message(mut self, message: Message) -> Self {
        self.messages.push(message);
        self
    }

    pub fn add_history(mut self, history: Vec<Message>) -> Self {
        self.messages.extend(history);
        self
    }

    pub fn add_user_text(mut self, text: impl Into<String>) -> Self {
        self.messages.push(Message::user(text.into()));
        self
    }

    pub fn add_model_text(mut self, text: impl Into<String>) -> Self {
        self.messages.push(Message::model(text.into()));
        self
    }

    /// Add a user message that includes a file handle reference.
    pub fn add_file(mut self, handle: Arc<FileHandle>, text: Option<String>) -> Result<Self> {
        let mut parts = vec![];
        if let Some(t) = text {
            parts.push(Part::Text {
                text: t,
                thought: None,
                thought_signature: None,
            });
        }
        parts.push(FileManager::as_part(&handle)?);

        let content = Content {
            parts: Some(parts),
            role: Some(Role::User),
        };
        self.messages.push(Message {
            role: Role::User,
            content,
        });
        Ok(self)
    }

    /// Add arbitrary parts as a user message.
    pub fn add_parts(mut self, parts: Vec<Part>) -> Self {
        let content = Content {
            parts: Some(parts),
            role: Some(Role::User),
        };
        self.messages.push(Message {
            role: Role::User,
            content,
        });
        self
    }

    /// Finalize into system instruction plus content list ready for `ContentBuilder`.
    pub fn build(self) -> (Option<String>, Vec<Content>) {
        let contents = self
            .messages
            .into_iter()
            .map(|m| m.content)
            .collect::<Vec<_>>();

        (self.system_instruction, contents)
    }
}

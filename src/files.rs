use std::{path::Path, sync::Arc};

use gemini_rust::{FileData, FileHandle, FileState, Gemini, Part};
use tokio::fs;
use tokio::time::{sleep, Duration};

use crate::error::{Result, StructuredError};

/// Helper for working with Gemini file handles.
#[derive(Clone)]
pub struct FileManager {
    client: Arc<Gemini>,
}

impl FileManager {
    pub fn new(client: Arc<Gemini>) -> Self {
        Self { client }
    }

    /// Upload a file from disk and return its handle.
    pub async fn upload_path<P: AsRef<Path>>(&self, path: P) -> Result<FileHandle> {
        let path_ref = path.as_ref();
        let bytes = fs::read(path_ref).await?;
        let mime = mime_guess::from_path(path_ref)
            .first_or_octet_stream()
            .to_string();
        let display_name = path_ref
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("uploaded_file")
            .to_string();

        self.upload_bytes(bytes, &mime, Some(&display_name)).await
    }

    /// Upload raw bytes with an explicit MIME type.
    pub async fn upload_bytes(
        &self,
        bytes: impl Into<Vec<u8>>,
        mime_type: &str,
        display_name: Option<&str>,
    ) -> Result<FileHandle> {
        let builder = self
            .client
            .create_file(bytes)
            .with_mime_type(mime_type.parse().map_err(|e| {
                StructuredError::Context(format!("Invalid MIME type '{mime_type}': {e}"))
            })?);

        let builder = if let Some(name) = display_name {
            builder.display_name(name.to_string())
        } else {
            builder
        };

        let handle = builder.upload().await?;
        Ok(handle)
    }

    /// Convert a handle into a `Part::FileData` usable by `ContextBuilder`.
    pub fn as_part(handle: &FileHandle) -> Result<Part> {
        let meta = handle.get_file_meta();
        let mime = meta
            .mime_type
            .as_ref()
            .ok_or_else(|| StructuredError::Context("file handle missing mime_type".to_string()))?
            .to_string();
        let uri = meta
            .uri
            .as_ref()
            .ok_or_else(|| StructuredError::Context("file handle missing uri".to_string()))?
            .to_string();

        Ok(Part::FileData {
            file_data: FileData {
                mime_type: mime,
                file_uri: uri,
            },
        })
    }

    /// Upload a file and wait for it to become active.
    pub async fn upload_and_wait<P: AsRef<Path>>(&self, path: P) -> Result<FileHandle> {
        let handle = self.upload_path(path).await?;
        self.wait_for_active(handle).await
    }

    /// Upload raw bytes and wait for the file to become active.
    pub async fn upload_bytes_and_wait(
        &self,
        bytes: impl Into<Vec<u8>>,
        mime_type: &str,
        display_name: Option<&str>,
    ) -> Result<FileHandle> {
        let handle = self.upload_bytes(bytes, mime_type, display_name).await?;
        self.wait_for_active(handle).await
    }

    async fn wait_for_active(&self, handle: FileHandle) -> Result<FileHandle> {
        let name = handle.name().to_string();
        let mut attempts = 0;
        loop {
            attempts += 1;
            let latest = self.client.get_file(&name).await?;
            if latest
                .get_file_meta()
                .state
                .as_ref()
                .is_some_and(|s| *s == FileState::Active)
            {
                return Ok(latest);
            }
            if attempts > 10 {
                return Err(StructuredError::Context(format!(
                    "file {name} not active after {attempts} checks"
                )));
            }
            sleep(Duration::from_secs(2)).await;
        }
    }
}

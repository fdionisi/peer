use anyhow::Result;
use async_trait::async_trait;

use crate::models::{
    conversation::{Conversation, ConversationId},
    message::{Message, MessageId},
    summary::Summary,
};

#[async_trait]
pub trait ConversationStore: Send + Sync {
    async fn current(&self) -> Result<ConversationId>;
    async fn set_current(&self, id: ConversationId) -> Result<()>;
    async fn create(&self, parent: Option<ConversationId>) -> Result<Conversation>;
    async fn append_message(&self, conversation_id: ConversationId, message: Message)
    -> Result<()>;
    async fn reference_messages(
        &self,
        conversation_id: ConversationId,
        message_ids: Vec<MessageId>,
    ) -> Result<()>;
    async fn update_summary(&self, conversation_id: ConversationId, summary: Summary)
    -> Result<()>;
    async fn get(&self, id: ConversationId) -> Result<Option<Conversation>>;
    async fn list_messages(&self, conversation_id: ConversationId) -> Result<Vec<Message>>;

    /// Atomically append a sequence of messages to a conversation.
    ///
    /// All messages are committed together or not at all. Use this instead of
    /// multiple `append_message` calls whenever the messages must appear as a
    /// unit — for example, a `ToolCall` immediately followed by its
    /// `ToolResult`.
    async fn append_messages(
        &self,
        conversation_id: ConversationId,
        messages: Vec<Message>,
    ) -> Result<()>;

    /// Atomically resolve a branch.
    ///
    /// Appends every message in `payload` to the branch's parent conversation,
    /// marks the branch as resolved, and moves the `current` pointer back to
    /// the parent — all in one operation. Either everything commits or nothing
    /// does, so the parent never holds a partial result.
    ///
    /// Fails if `child_id` has no parent, or if it is already resolved.
    async fn resolve_branch(&self, child_id: ConversationId, payload: Vec<Message>) -> Result<()>;
}

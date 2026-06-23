use jiff::Timestamp;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::models::{message::MessageId, summary::Summary};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConversationId(Uuid);

impl std::fmt::Display for ConversationId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConversationItem {
    Message(MessageId),
    ReferencedMessage(MessageId),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub id: ConversationId,
    pub parent: Option<ConversationId>,
    pub items: Vec<ConversationItem>,
    pub summary: Option<Summary>,
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
    /// Set when this branch has been resolved. `None` means the branch is
    /// still open. A resolved branch has delivered its payload to the parent
    /// and handed current back; it will not receive further turns.
    pub resolved_at: Option<Timestamp>,
}

impl ConversationId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }

    pub fn as_uuid(&self) -> &Uuid {
        &self.0
    }
}

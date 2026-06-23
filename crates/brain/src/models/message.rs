use jiff::Timestamp;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MessageId(Uuid);

impl std::fmt::Display for MessageId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: MessageId,
    pub role: Role,
    pub content: Vec<Content>,
    pub timestamp: Timestamp,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Role {
    User,
    Assistant,
    System,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Content {
    Text {
        text: String,
    },
    ToolCall {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        id: String,
        output: String,
        is_error: bool,
    },
    /// A hidden temporal context message injected by the orchestrator.
    /// Never displayed to the user. Tells the model the current timestamp
    /// so it can reason about time without guessing.
    TemporalUpdate {
        timestamp: String,
    },
}

impl MessageId {
    pub fn new() -> Self {
        Self(Uuid::now_v7())
    }

    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }

    pub fn as_uuid(&self) -> &Uuid {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_round_trips_through_serde_json() {
        let original = vec![
            Content::Text {
                text: "hello".to_string(),
            },
            Content::ToolCall {
                id: "call-1".to_string(),
                name: "web_search".to_string(),
                input: serde_json::json!({ "query": "rust async traits" }),
            },
            Content::ToolResult {
                id: "call-1".to_string(),
                output: "first hit".to_string(),
                is_error: false,
            },
        ];

        let json = serde_json::to_string(&original).expect("serialize");
        let decoded: Vec<Content> = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(decoded, original);
    }
}

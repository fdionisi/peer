use std::collections::HashMap;
use std::sync::Mutex;

use anyhow::Result;
use async_trait::async_trait;

use crate::models::{
    conversation::{Conversation, ConversationId, ConversationItem},
    message::{Message, MessageId},
    summary::Summary,
};

pub struct InMemoryConversationStore {
    inner: Mutex<Inner>,
}

struct Inner {
    conversations: HashMap<ConversationId, Conversation>,
    messages: HashMap<MessageId, Message>,
    current: Option<ConversationId>,
}

impl InMemoryConversationStore {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Inner {
                conversations: HashMap::new(),
                messages: HashMap::new(),
                current: None,
            }),
        }
    }

    pub fn add_conversation(&self, conversation: Conversation) {
        let mut inner = self.inner.lock().unwrap();
        let id = conversation.id;
        if inner.current.is_none() {
            inner.current = Some(id);
        }
        inner.conversations.insert(id, conversation);
    }

    pub fn add_referenced_message(&self, conversation_id: ConversationId, message: Message) {
        let mut inner = self.inner.lock().unwrap();
        let id = message.id;
        inner.messages.insert(id, message);
        if let Some(conv) = inner.conversations.get_mut(&conversation_id) {
            conv.items.push(ConversationItem::ReferencedMessage(id));
        }
    }

    pub fn get_conversation(&self, id: ConversationId) -> Option<Conversation> {
        let inner = self.inner.lock().unwrap();
        inner.conversations.get(&id).cloned()
    }

    pub fn get_message(&self, id: MessageId) -> Option<Message> {
        let inner = self.inner.lock().unwrap();
        inner.messages.get(&id).cloned()
    }
}

impl Default for InMemoryConversationStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl crate::conversation_store::ConversationStore for InMemoryConversationStore {
    async fn current(&self) -> Result<ConversationId> {
        let inner = self.inner.lock().unwrap();
        inner
            .current
            .ok_or_else(|| anyhow::anyhow!("no current conversation"))
    }

    async fn set_current(&self, id: ConversationId) -> Result<()> {
        let mut inner = self.inner.lock().unwrap();
        inner.current = Some(id);
        Ok(())
    }

    async fn create(&self, parent: Option<ConversationId>) -> Result<Conversation> {
        let id = ConversationId::new();
        let now = jiff::Timestamp::now();
        let conversation = Conversation {
            id,
            parent,
            items: Vec::new(),
            summary: None,
            created_at: now,
            updated_at: now,
            resolved_at: None,
        };
        let mut inner = self.inner.lock().unwrap();
        inner
            .conversations
            .insert(conversation.id, conversation.clone());
        Ok(conversation)
    }

    async fn append_message(
        &self,
        conversation_id: ConversationId,
        message: Message,
    ) -> Result<()> {
        let mut inner = self.inner.lock().unwrap();
        let id = message.id;
        inner.messages.insert(id, message);
        if let Some(conv) = inner.conversations.get_mut(&conversation_id) {
            conv.items.push(ConversationItem::Message(id));
            conv.updated_at = jiff::Timestamp::now();
        }
        Ok(())
    }

    async fn reference_messages(
        &self,
        conversation_id: ConversationId,
        message_ids: Vec<MessageId>,
    ) -> Result<()> {
        let mut inner = self.inner.lock().unwrap();
        if let Some(conv) = inner.conversations.get_mut(&conversation_id) {
            for id in message_ids {
                conv.items.push(ConversationItem::ReferencedMessage(id));
            }
            conv.updated_at = jiff::Timestamp::now();
        }
        Ok(())
    }

    async fn update_summary(
        &self,
        conversation_id: ConversationId,
        summary: Summary,
    ) -> Result<()> {
        let mut inner = self.inner.lock().unwrap();
        if let Some(conv) = inner.conversations.get_mut(&conversation_id) {
            conv.summary = Some(summary);
            conv.updated_at = jiff::Timestamp::now();
        }
        Ok(())
    }

    async fn get(&self, id: ConversationId) -> Result<Option<Conversation>> {
        let inner = self.inner.lock().unwrap();
        Ok(inner.conversations.get(&id).cloned())
    }

    async fn list_messages(&self, conversation_id: ConversationId) -> Result<Vec<Message>> {
        let inner = self.inner.lock().unwrap();
        let conv = match inner.conversations.get(&conversation_id) {
            Some(c) => c,
            None => return Ok(Vec::new()),
        };
        let messages: Vec<Message> = conv
            .items
            .iter()
            .filter_map(|item| match item {
                ConversationItem::Message(id) | ConversationItem::ReferencedMessage(id) => {
                    inner.messages.get(id).cloned()
                }
            })
            .collect();
        Ok(messages)
    }

    async fn append_messages(
        &self,
        conversation_id: ConversationId,
        messages: Vec<Message>,
    ) -> Result<()> {
        let mut inner = self.inner.lock().unwrap();
        let now = jiff::Timestamp::now();
        for message in messages {
            let id = message.id;
            inner.messages.insert(id, message);
            if let Some(conv) = inner.conversations.get_mut(&conversation_id) {
                conv.items.push(ConversationItem::Message(id));
                conv.updated_at = now;
            }
        }
        Ok(())
    }

    async fn resolve_branch(&self, child_id: ConversationId, payload: Vec<Message>) -> Result<()> {
        let mut inner = self.inner.lock().unwrap();

        let parent_id = inner
            .conversations
            .get(&child_id)
            .and_then(|c| c.parent)
            .ok_or_else(|| anyhow::anyhow!("child {child_id} has no parent"))?;

        if inner
            .conversations
            .get(&child_id)
            .map(|c| c.resolved_at.is_some())
            .unwrap_or(false)
        {
            return Err(anyhow::anyhow!("branch {child_id} is already resolved"));
        }

        let now = jiff::Timestamp::now();
        for message in payload {
            let id = message.id;
            inner.messages.insert(id, message);
            if let Some(conv) = inner.conversations.get_mut(&parent_id) {
                conv.items.push(ConversationItem::Message(id));
                conv.updated_at = now;
            }
        }

        if let Some(child) = inner.conversations.get_mut(&child_id) {
            child.resolved_at = Some(now);
            child.updated_at = now;
        }

        inner.current = Some(parent_id);

        Ok(())
    }
}

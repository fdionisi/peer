use anyhow::{Context, Result};
use async_trait::async_trait;
use brain::{
    conversation_store::ConversationStore,
    embedder::Embedder,
    memory_index::{MemoryIndex, RelatedConversation},
    models::{
        conversation::{Conversation, ConversationId, ConversationItem},
        message::{Content, Message, MessageId, Role},
        summary::Summary,
    },
    tool::{ToolRegistry, Visibility},
    tool_index::{DiscoveredTool, ToolIndex},
};
use jiff::Timestamp;
use serde::{Deserialize, Serialize};
use surrealdb::{Surreal, engine::any::Any};
use surrealdb_types::{RecordId, RecordIdKey, SurrealValue};

use crate::migrations::MigrationRunner;

/// Embedding vector dimension used by the HNSW index.
///
/// Must match the `HNSW DIMENSION` value in the recall migration and the
/// dimension of vectors produced by the configured `Embedder`.
pub const EMBED_DIMENSION: usize = 1024;

pub struct SurrealDbClient {
    db: Surreal<Any>,
}

impl SurrealDbClient {
    pub fn new(db: Surreal<Any>) -> Self {
        Self { db }
    }

    pub async fn migrate(&self, migrations_dir: std::path::PathBuf) -> Result<()> {
        MigrationRunner::new(self.db.clone(), migrations_dir)
            .run()
            .await
    }

    fn conversation_record_id(id: ConversationId) -> RecordId {
        RecordId {
            table: "conversation".into(),
            key: RecordIdKey::String(id.as_uuid().to_string()),
        }
    }

    fn message_record_id(id: MessageId) -> RecordId {
        RecordId {
            table: "message".into(),
            key: RecordIdKey::String(id.as_uuid().to_string()),
        }
    }

    fn parse_conversation_id(record_id: &RecordId) -> Result<ConversationId> {
        let key = match &record_id.key {
            RecordIdKey::String(s) => s,
            _ => return Err(anyhow::anyhow!("invalid conversation record key")),
        };
        let uuid = uuid::Uuid::parse_str(key).context("invalid conversation uuid")?;
        Ok(ConversationId::from_uuid(uuid))
    }

    fn parse_message_id(record_id: &RecordId) -> Result<MessageId> {
        let key = match &record_id.key {
            RecordIdKey::String(s) => s,
            _ => return Err(anyhow::anyhow!("invalid message record key")),
        };
        let uuid = uuid::Uuid::parse_str(key).context("invalid message uuid")?;
        Ok(MessageId::from_uuid(uuid))
    }

    fn timestamp_to_string(ts: Timestamp) -> String {
        ts.to_string()
    }

    fn timestamp_from_string(s: &str) -> Result<Timestamp> {
        use std::str::FromStr;
        Timestamp::from_str(s).context("invalid timestamp")
    }

    /// Escape a string value for embedding in a single-quoted SurrealQL string
    /// literal. SurrealQL treats `\` as an escape character inside `'...'`, so
    /// we must escape backslashes first, then single quotes.
    fn surreal_escape(s: &str) -> String {
        s.replace('\\', "\\\\").replace('\'', "\\'")
    }
}

#[derive(Debug, Serialize, Deserialize, SurrealValue)]
struct ConversationRecord {
    #[serde(default)]
    id: Option<RecordId>,
    parent: Option<RecordId>,
    summary_content: Option<String>,
    summary_created_at: Option<String>,
    created_at: String,
    updated_at: String,
    #[serde(default)]
    resolved_at: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, SurrealValue)]
struct MessageRecord {
    #[serde(default)]
    id: Option<RecordId>,
    role: String,
    content: String,
    timestamp: String,
}

#[derive(Debug, Serialize, Deserialize, SurrealValue)]
struct CurrentRecord {
    current_id: Option<RecordId>,
    #[serde(default)]
    updated_at: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, SurrealValue)]
struct ItemRelation {
    out: RecordId,
    item_type: String,
}

#[async_trait]
impl ConversationStore for SurrealDbClient {
    async fn current(&self) -> Result<ConversationId> {
        let mut response = self
            .db
            .query("SELECT current_id FROM current_conversation:singleton")
            .await?;
        let rows: Vec<CurrentRecord> = response.take(0)?;
        let record = rows
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("current_conversation singleton not found"))?;
        let current_id = record
            .current_id
            .ok_or_else(|| anyhow::anyhow!("no current conversation set"))?;
        Self::parse_conversation_id(&current_id)
    }

    async fn set_current(&self, id: ConversationId) -> Result<()> {
        let now = Self::timestamp_to_string(Timestamp::now());
        let record_id = Self::conversation_record_id(id);
        self.db
            .query(
                r#"
                UPSERT current_conversation:singleton SET
                    current_id = $record_id,
                    updated_at = $now
                "#,
            )
            .bind(("record_id", record_id))
            .bind(("now", now))
            .await?;
        Ok(())
    }

    async fn create(&self, parent: Option<ConversationId>) -> Result<Conversation> {
        let id = ConversationId::new();
        let now_ts = Timestamp::now();
        let now = Self::timestamp_to_string(now_ts);
        let record_id = Self::conversation_record_id(id);

        let parent_record_id = parent.map(Self::conversation_record_id);

        self.db
            .query(
                r#"
                CREATE $record_id SET
                    parent = $parent,
                    summary_content = NONE,
                    summary_created_at = NONE,
                    resolved_at = NONE,
                    created_at = $now,
                    updated_at = $now
                "#,
            )
            .bind(("record_id", record_id))
            .bind(("parent", parent_record_id))
            .bind(("now", now))
            .await?;

        Ok(Conversation {
            id,
            parent,
            items: Vec::new(),
            summary: None,
            created_at: now_ts,
            updated_at: now_ts,
            resolved_at: None,
        })
    }

    async fn append_message(
        &self,
        conversation_id: ConversationId,
        message: Message,
    ) -> Result<()> {
        let conv_record_id = Self::conversation_record_id(conversation_id);
        let msg_id = message.id;
        let msg_record_id = Self::message_record_id(msg_id);

        let role_str = match message.role {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::System => "system",
        };

        let content_json =
            serde_json::to_string(&message.content).context("failed to serialize content")?;
        let timestamp = Self::timestamp_to_string(message.timestamp);

        self.db
            .query(
                r#"
                    CREATE $msg_record_id SET
                        role = $role,
                        content = $content,
                        timestamp = $timestamp
                    "#,
            )
            .bind(("msg_record_id", msg_record_id))
            .bind(("role", role_str))
            .bind(("content", content_json))
            .bind(("timestamp", timestamp))
            .await?;

        self.db
            .query(
                r#"
                    RELATE $conv_record_id->conversation_item->$msg_record_id SET
                        item_type = 'message'
                    "#,
            )
            .bind(("conv_record_id", conv_record_id))
            .bind(("msg_record_id", Self::message_record_id(msg_id)))
            .await?;

        Ok(())
    }

    async fn reference_messages(
        &self,
        conversation_id: ConversationId,
        message_ids: Vec<MessageId>,
    ) -> Result<()> {
        let conv_record_id = Self::conversation_record_id(conversation_id);

        for msg_id in message_ids.iter() {
            self.db
                .query(
                    r#"
                        RELATE $conv_record_id->conversation_item->$msg_record_id SET
                            item_type = 'referenced'
                        "#,
                )
                .bind(("conv_record_id", conv_record_id.clone()))
                .bind(("msg_record_id", Self::message_record_id(*msg_id)))
                .await?;
        }

        Ok(())
    }

    async fn update_summary(
        &self,
        conversation_id: ConversationId,
        summary: Summary,
    ) -> Result<()> {
        let record_id = Self::conversation_record_id(conversation_id);
        let now = Self::timestamp_to_string(Timestamp::now());
        let created_at = Self::timestamp_to_string(summary.created_at);
        self.db
            .query(
                r#"
                UPDATE $record_id SET
                    summary_content = $content,
                    summary_created_at = $created_at,
                    updated_at = $now
                "#,
            )
            .bind(("record_id", record_id))
            .bind(("content", summary.content))
            .bind(("created_at", created_at))
            .bind(("now", now))
            .await?;
        Ok(())
    }

    async fn get(&self, id: ConversationId) -> Result<Option<Conversation>> {
        let record_id = Self::conversation_record_id(id);
        let mut response = self
            .db
            .query("SELECT * FROM $record_id")
            .bind(("record_id", record_id))
            .await?;
        let rows: Vec<ConversationRecord> = response.take(0)?;
        let Some(record) = rows.into_iter().next() else {
            return Ok(None);
        };

        let parent = record
            .parent
            .map(|p| Self::parse_conversation_id(&p))
            .transpose()?;

        let summary = match (record.summary_content, record.summary_created_at) {
            (Some(content), Some(created_at_str)) => {
                let created_at = Self::timestamp_from_string(&created_at_str)?;
                Some(Summary {
                    content,
                    created_at,
                })
            }
            _ => None,
        };

        let items = self.list_items(id).await?;

        let resolved_at = record
            .resolved_at
            .as_deref()
            .map(Self::timestamp_from_string)
            .transpose()?;

        Ok(Some(Conversation {
            id,
            parent,
            items,
            summary,
            created_at: Self::timestamp_from_string(&record.created_at)?,
            updated_at: Self::timestamp_from_string(&record.updated_at)?,
            resolved_at,
        }))
    }

    async fn list_messages(&self, conversation_id: ConversationId) -> Result<Vec<Message>> {
        let conv_record_id = Self::conversation_record_id(conversation_id);

        let mut response = self
            .db
            .query(
                r#"
                    SELECT out, item_type FROM conversation_item
                    WHERE `in` = $conv_record_id
                    "#,
            )
            .bind(("conv_record_id", conv_record_id))
            .await?;
        let rows: Vec<ItemRelation> = response.take(0)?;

        let mut items: Vec<(MessageId, String)> = rows
            .into_iter()
            .map(|r| Ok((Self::parse_message_id(&r.out)?, r.item_type)))
            .collect::<Result<Vec<_>>>()?;
        items.sort_by_key(|(id, _)| *id.as_uuid());

        let mut messages = Vec::new();
        for (msg_id, _item_type) in items {
            let mut msg_response = self
                .db
                .query("SELECT * FROM $msg_record_id")
                .bind(("msg_record_id", Self::message_record_id(msg_id)))
                .await?;
            let msg_rows: Vec<MessageRecord> = msg_response.take(0)?;
            let Some(msg_record) = msg_rows.into_iter().next() else {
                continue;
            };

            let role = match msg_record.role.as_str() {
                "user" => Role::User,
                "assistant" => Role::Assistant,
                "system" => Role::System,
                _ => Role::User,
            };

            let content: Vec<Content> = serde_json::from_str(&msg_record.content)
                .context("failed to deserialize content")?;

            messages.push(Message {
                id: msg_id,
                role,
                content,
                timestamp: Self::timestamp_from_string(&msg_record.timestamp)?,
            });
        }

        Ok(messages)
    }

    async fn append_messages(
        &self,
        conversation_id: ConversationId,
        messages: Vec<Message>,
    ) -> Result<()> {
        if messages.is_empty() {
            return Ok(());
        }

        let conv_uuid = conversation_id.as_uuid();
        let now = Self::timestamp_to_string(Timestamp::now());
        let mut sql = String::from("BEGIN TRANSACTION;\n");

        for message in &messages {
            let msg_id = message.id.as_uuid();
            let role_str = match message.role {
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::System => "system",
            };
            let content_json =
                serde_json::to_string(&message.content).context("failed to serialize content")?;
            let content_escaped = Self::surreal_escape(&content_json);
            let timestamp = Self::timestamp_to_string(message.timestamp);
            sql.push_str(&format!(
                "CREATE message:`{msg_id}` SET role = '{role_str}', content = '{content_escaped}', timestamp = '{timestamp}';\n"
            ));
            sql.push_str(&format!(
                "RELATE conversation:`{conv_uuid}`->conversation_item->message:`{msg_id}` SET item_type = 'message';\n"
            ));
        }

        sql.push_str(&format!(
            "UPDATE conversation:`{conv_uuid}` SET updated_at = '{now}';\n"
        ));
        sql.push_str("COMMIT TRANSACTION;");

        self.db
            .query(sql)
            .await
            .context("append_messages transaction failed")?;

        Ok(())
    }

    async fn resolve_branch(&self, child_id: ConversationId, payload: Vec<Message>) -> Result<()> {
        let child = self
            .get(child_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("branch {child_id} not found"))?;

        let parent_id = child
            .parent
            .ok_or_else(|| anyhow::anyhow!("branch {child_id} has no parent"))?;

        if child.resolved_at.is_some() {
            return Err(anyhow::anyhow!("branch {child_id} is already resolved"));
        }

        let now = Self::timestamp_to_string(Timestamp::now());

        let mut sql = String::from("BEGIN TRANSACTION;\n");

        for message in &payload {
            let msg_id = message.id.as_uuid();
            let role_str = match message.role {
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::System => "system",
            };
            let content_json =
                serde_json::to_string(&message.content).context("failed to serialize content")?;
            let timestamp = Self::timestamp_to_string(message.timestamp);
            let content_escaped = Self::surreal_escape(&content_json);
            sql.push_str(&format!(
                "CREATE message:`{msg_id}` SET role = '{role_str}', content = '{content_escaped}', timestamp = '{timestamp}';\n"
            ));
            sql.push_str(&format!(
                "RELATE conversation:`{parent_record_id_key}`->conversation_item->message:`{msg_id}` SET item_type = 'message';\n",
                parent_record_id_key = parent_id.as_uuid()
            ));
        }

        let parent_uuid = parent_id.as_uuid();
        let child_uuid = child_id.as_uuid();

        sql.push_str(&format!(
            "UPDATE conversation:`{parent_uuid}` SET updated_at = '{now}';\n"
        ));
        sql.push_str(&format!(
            "UPDATE conversation:`{child_uuid}` SET resolved_at = '{now}', updated_at = '{now}';\n"
        ));
        sql.push_str(&format!(
            "UPSERT current_conversation:singleton SET current_id = conversation:`{parent_uuid}`, updated_at = '{now}';\n"
        ));
        sql.push_str("COMMIT TRANSACTION;");

        self.db
            .query(sql)
            .await
            .context("resolve_branch transaction failed")?;

        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize, SurrealValue)]
struct RecallSearchRow {
    id: RecordId,
    score: f32,
}

#[derive(Debug, Serialize, Deserialize, SurrealValue)]
struct RecallEdgeRow {
    out: RecordId,
    #[serde(default)]
    created_at: Option<String>,
}

#[async_trait]
impl MemoryIndex for SurrealDbClient {
    async fn index_summary(&self, id: ConversationId, embedding: Vec<f32>) -> Result<()> {
        let record_id = Self::conversation_record_id(id);
        self.db
            .query("UPDATE $id SET summary_embedding = $embedding")
            .bind(("id", record_id))
            .bind(("embedding", embedding))
            .await?;
        Ok(())
    }

    async fn search(
        &self,
        query: Vec<f32>,
        k: usize,
        exclude: Option<ConversationId>,
    ) -> Result<Vec<RelatedConversation>> {
        let (surql, exclude_record) = match exclude {
            Some(excl_id) => (
                format!(
                    r#"SELECT id, vector::similarity::cosine(summary_embedding, $query) AS score
                               FROM conversation
                               WHERE summary_embedding IS NOT NULL
                                 AND summary_embedding <|{k}, 150|> $query
                                 AND id != $exclude
                               ORDER BY score DESC"#
                ),
                Some(Self::conversation_record_id(excl_id)),
            ),
            None => (
                format!(
                    r#"SELECT id, vector::similarity::cosine(summary_embedding, $query) AS score
                               FROM conversation
                               WHERE summary_embedding IS NOT NULL
                                 AND summary_embedding <|{k}, 150|> $query
                               ORDER BY score DESC"#
                ),
                None,
            ),
        };

        let mut db_query = self.db.query(&surql).bind(("query", query));
        if let Some(excl) = exclude_record {
            db_query = db_query.bind(("exclude", excl));
        }

        let mut response = db_query.await?;
        let rows: Vec<RecallSearchRow> = response.take(0)?;

        rows.into_iter()
            .map(|r| {
                let id = Self::parse_conversation_id(&r.id)?;
                Ok(RelatedConversation { id, score: r.score })
            })
            .collect()
    }

    async fn record_recall(
        &self,
        from: ConversationId,
        to: ConversationId,
        score: f32,
    ) -> Result<()> {
        let from_id = Self::conversation_record_id(from);
        let to_id = Self::conversation_record_id(to);
        let now = Self::timestamp_to_string(Timestamp::now());
        self.db
            .query(
                r#"
                DELETE related WHERE `in` = $from AND out = $to AND kind = 'recall';
                RELATE $from->related->$to SET kind = 'recall', score = $score, created_at = $now
                "#,
            )
            .bind(("from", from_id))
            .bind(("to", to_id))
            .bind(("score", score))
            .bind(("now", now))
            .await?;
        Ok(())
    }

    async fn recalled(&self, from: ConversationId) -> Result<Vec<ConversationId>> {
        let from_id = Self::conversation_record_id(from);
        let mut response = self
                .db
                .query(
                    "SELECT out, created_at FROM related WHERE `in` = $from AND kind = 'recall' ORDER BY created_at ASC",
                )
                .bind(("from", from_id))
                .await?;
        let rows: Vec<RecallEdgeRow> = response.take(0)?;
        rows.into_iter()
            .map(|r| Self::parse_conversation_id(&r.out))
            .collect()
    }
}

impl SurrealDbClient {
    async fn list_items(&self, conversation_id: ConversationId) -> Result<Vec<ConversationItem>> {
        let conv_record_id = Self::conversation_record_id(conversation_id);

        let mut response = self
            .db
            .query(
                r#"
                    SELECT out, item_type FROM conversation_item
                    WHERE `in` = $conv_record_id
                    "#,
            )
            .bind(("conv_record_id", conv_record_id))
            .await?;
        let rows: Vec<ItemRelation> = response.take(0)?;

        let mut items: Vec<(MessageId, String)> = rows
            .into_iter()
            .map(|r| Ok((Self::parse_message_id(&r.out)?, r.item_type)))
            .collect::<Result<Vec<_>>>()?;
        items.sort_by_key(|(id, _)| *id.as_uuid());

        let mut result = Vec::new();
        for (msg_id, item_type) in items {
            result.push(match item_type.as_str() {
                "referenced" => ConversationItem::ReferencedMessage(msg_id),
                _ => ConversationItem::Message(msg_id),
            });
        }

        Ok(result)
    }
}

fn tool_embedding_text(name: &str, description: &str) -> String {
    format!("{}\n{}", name, description)
}

#[derive(Debug, Serialize, Deserialize, SurrealValue)]
struct ToolRow {
    name: String,
    #[serde(default)]
    score: Option<f32>,
}

#[async_trait]
impl ToolIndex for SurrealDbClient {
    async fn index(&self, registry: &dyn ToolRegistry, embedder: &dyn Embedder) -> Result<()> {
        self.db.query("DELETE tool").await?;

        for definition in registry.definitions() {
            if registry.visibility(&definition.name) != Some(Visibility::Hidden) {
                continue;
            }

            let text = tool_embedding_text(&definition.name, &definition.description);
            let embedding = embedder.embed(&text).await?;

            self.db
                .query(
                    r#"
                    CREATE tool SET
                        name = $name,
                        embedding = $embedding
                    "#,
                )
                .bind(("name", definition.name.clone()))
                .bind(("embedding", embedding))
                .await?;
        }

        Ok(())
    }

    async fn search(&self, query: Vec<f32>, k: usize) -> Result<Vec<DiscoveredTool>> {
        if k == 0 {
            return Ok(Vec::new());
        }

        let surql = format!(
            r#"
            SELECT name,
                   vector::similarity::cosine(embedding, $query) AS score
            FROM tool
            WHERE embedding IS NOT NULL
              AND embedding <|{k}, 150|> $query
            ORDER BY score DESC
            "#
        );

        let mut response = self.db.query(&surql).bind(("query", query)).await?;
        let rows: Vec<ToolRow> = response.take(0)?;

        rows.into_iter()
            .map(|r| {
                let score = r.score.unwrap_or(0.0).clamp(0.0, 1.0);
                Ok(DiscoveredTool {
                    name: r.name,
                    score,
                })
            })
            .collect()
    }
}

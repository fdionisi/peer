use std::sync::Arc;

use brain::{
    conversation_store::ConversationStore,
    embedder::Embedder,
    memory_index::MemoryIndex,
    models::{
        message::{Content, Message, MessageId, Role},
        summary::Summary,
    },
    testing::tool_registry::MockToolRegistry,
    tool::{Policy, ToolOutput},
    tool_index::ToolIndex,
};
use brain_surrealdb::{EMBED_DIMENSION, SurrealDbClient, default_migrations_dir};
use jiff::Timestamp;
use surrealdb::{Surreal, engine::any::Any};
use testcontainers::{
    ContainerAsync, GenericImage, ImageExt,
    core::{ContainerPort, WaitFor},
    runners::AsyncRunner,
};

struct TestStore {
    _container: ContainerAsync<GenericImage>,
    store: Arc<SurrealDbClient>,
}

impl TestStore {
    fn store(&self) -> &Arc<SurrealDbClient> {
        &self.store
    }
}

async fn setup() -> TestStore {
    let container = GenericImage::new("surrealdb/surrealdb", "v3.1.5")
        .with_exposed_port(ContainerPort::Tcp(8000))
        .with_wait_for(WaitFor::seconds(5))
        .with_cmd(["start", "--user", "root", "--pass", "root"])
        .start()
        .await
        .unwrap();
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    let port = container.get_host_port_ipv4(8000).await.unwrap();

    let db: Surreal<Any> = Surreal::init();
    db.connect(format!("ws://127.0.0.1:{port}")).await.unwrap();
    db.signin(surrealdb::opt::auth::Root {
        username: "root".to_string(),
        password: "root".to_string(),
    })
    .await
    .unwrap();

    let ns = format!("test_{}", uuid::Uuid::new_v4().simple());
    db.use_ns(&ns).use_db("test").await.unwrap();

    let store = Arc::new(SurrealDbClient::new(db));
    store.migrate(default_migrations_dir()).await.unwrap();

    TestStore {
        _container: container,
        store,
    }
}

fn make_message(role: Role, text: &str) -> Message {
    Message {
        id: MessageId::new(),
        role,
        content: vec![Content::Text {
            text: text.to_string(),
        }],
        timestamp: Timestamp::now(),
    }
}

#[tokio::test]
async fn create_and_get_conversation() {
    let test = setup().await;

    let conv = test.store().create(None).await.unwrap();
    let fetched = test.store().get(conv.id).await.unwrap();

    assert!(fetched.is_some());
    let fetched = fetched.unwrap();
    assert_eq!(fetched.id, conv.id);
    assert!(fetched.parent.is_none());
    assert!(fetched.summary.is_none());
    assert!(fetched.items.is_empty());
}

#[tokio::test]
async fn create_child_conversation() {
    let test = setup().await;

    let parent = test.store().create(None).await.unwrap();
    let child = test.store().create(Some(parent.id)).await.unwrap();

    let fetched_parent = test.store().get(parent.id).await.unwrap().unwrap();
    let fetched_child = test.store().get(child.id).await.unwrap().unwrap();

    assert_eq!(fetched_child.parent, Some(parent.id));
    assert!(fetched_parent.parent.is_none());
}

#[tokio::test]
async fn append_and_list_messages() {
    let test = setup().await;

    let conv = test.store().create(None).await.unwrap();
    let msg1 = make_message(Role::User, "hello");
    let msg2 = make_message(Role::Assistant, "hi");

    test.store()
        .append_message(conv.id, msg1.clone())
        .await
        .unwrap();
    test.store()
        .append_message(conv.id, msg2.clone())
        .await
        .unwrap();

    let messages = test.store().list_messages(conv.id).await.unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].id, msg1.id);
    assert_eq!(messages[1].id, msg2.id);
}

#[tokio::test]
async fn reference_messages() {
    let test = setup().await;

    let parent = test.store().create(None).await.unwrap();
    let msg1 = make_message(Role::User, "hello");
    let msg2 = make_message(Role::Assistant, "hi");

    test.store()
        .append_message(parent.id, msg1.clone())
        .await
        .unwrap();
    test.store()
        .append_message(parent.id, msg2.clone())
        .await
        .unwrap();

    let child = test.store().create(Some(parent.id)).await.unwrap();
    test.store()
        .reference_messages(child.id, vec![msg1.id, msg2.id])
        .await
        .unwrap();

    let child_messages = test.store().list_messages(child.id).await.unwrap();
    assert_eq!(child_messages.len(), 2);
    assert_eq!(child_messages[0].id, msg1.id);
    assert_eq!(child_messages[1].id, msg2.id);

    let child_items = test.store().get(child.id).await.unwrap().unwrap().items;
    assert_eq!(child_items.len(), 2);
    assert!(matches!(
        child_items[0],
        brain::models::conversation::ConversationItem::ReferencedMessage(_)
    ));
}

#[tokio::test]
async fn update_and_get_summary() {
    let test = setup().await;

    let conv = test.store().create(None).await.unwrap();
    let summary = Summary {
        content: "Earlier we discussed Rust".to_string(),
        created_at: Timestamp::now(),
    };

    test.store()
        .update_summary(conv.id, summary.clone())
        .await
        .unwrap();

    let fetched = test.store().get(conv.id).await.unwrap().unwrap();
    assert!(fetched.summary.is_some());
    let fetched_summary = fetched.summary.unwrap();
    assert_eq!(fetched_summary.content, summary.content);
}

#[tokio::test]
async fn current_conversation_pointer() {
    let test = setup().await;

    let conv = test.store().create(None).await.unwrap();
    test.store().set_current(conv.id).await.unwrap();

    let current = test.store().current().await.unwrap();
    assert_eq!(current, conv.id);
}

#[tokio::test]
async fn migrations_are_idempotent() {
    let test = setup().await;
    test.store()
        .migrate(default_migrations_dir())
        .await
        .unwrap();
    test.store()
        .migrate(default_migrations_dir())
        .await
        .unwrap();
}

#[tokio::test]
async fn messages_are_returned_in_chronological_order() {
    let test = setup().await;

    let conv = test.store().create(None).await.unwrap();
    let msg1 = make_message(Role::User, "first");
    let msg2 = make_message(Role::Assistant, "second");
    let msg3 = make_message(Role::User, "third");

    test.store()
        .append_message(conv.id, msg1.clone())
        .await
        .unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(2)).await;
    test.store()
        .append_message(conv.id, msg2.clone())
        .await
        .unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(2)).await;
    test.store()
        .append_message(conv.id, msg3.clone())
        .await
        .unwrap();

    let messages = test.store().list_messages(conv.id).await.unwrap();
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0].id, msg1.id);
    assert_eq!(messages[1].id, msg2.id);
    assert_eq!(messages[2].id, msg3.id);
}

#[tokio::test]
async fn referenced_messages_preserve_insertion_order() {
    let test = setup().await;

    let parent = test.store().create(None).await.unwrap();
    let msg1 = make_message(Role::User, "first");
    let msg2 = make_message(Role::Assistant, "second");
    let msg3 = make_message(Role::User, "third");

    test.store()
        .append_message(parent.id, msg1.clone())
        .await
        .unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(2)).await;
    test.store()
        .append_message(parent.id, msg2.clone())
        .await
        .unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(2)).await;
    test.store()
        .append_message(parent.id, msg3.clone())
        .await
        .unwrap();

    let child = test.store().create(Some(parent.id)).await.unwrap();
    test.store()
        .reference_messages(child.id, vec![msg1.id, msg2.id, msg3.id])
        .await
        .unwrap();

    let messages = test.store().list_messages(child.id).await.unwrap();
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0].id, msg1.id);
    assert_eq!(messages[1].id, msg2.id);
    assert_eq!(messages[2].id, msg3.id);
}

// ── MemoryIndex tests ────────────────────────────────────────────────────────

/// Returns a 1024-dim unit vector with a single hot dimension.
/// Two vectors with different hot dimensions are orthogonal (cosine = 0);
/// the same hot dimension gives cosine = 1.
fn unit_vec(hot: usize) -> Vec<f32> {
    assert!(hot < EMBED_DIMENSION, "hot index out of range");
    let mut v = vec![0.0f32; EMBED_DIMENSION];
    v[hot] = 1.0;
    v
}

#[tokio::test]
async fn memory_index_search_finds_indexed_conversation() {
    let test = setup().await;

    let conv = test.store().create(None).await.unwrap();
    let embedding = unit_vec(0);

    test.store()
        .index_summary(conv.id, embedding.clone())
        .await
        .unwrap();

    let results = MemoryIndex::search(test.store().as_ref(), embedding, 5, None)
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, conv.id);
    // Identical vectors have cosine similarity 1.0.
    assert!(
        (results[0].score - 1.0).abs() < 1e-4,
        "expected score ~1.0, got {}",
        results[0].score
    );
}

#[tokio::test]
async fn memory_index_nearer_vector_outranks_farther() {
    let test = setup().await;

    let conv_a = test.store().create(None).await.unwrap();
    let conv_b = test.store().create(None).await.unwrap();

    // conv_a aligned with dimension 0, conv_b aligned with dimension 1.
    test.store()
        .index_summary(conv_a.id, unit_vec(0))
        .await
        .unwrap();
    test.store()
        .index_summary(conv_b.id, unit_vec(1))
        .await
        .unwrap();

    // Query on dimension 0 — conv_a should be the best match.
    let results = MemoryIndex::search(test.store().as_ref(), unit_vec(0), 5, None)
        .await
        .unwrap();
    assert!(
        results.len() >= 2,
        "expected both conversations, got {}",
        results.len()
    );
    assert_eq!(results[0].id, conv_a.id, "closer vector should rank first");
    assert!(
        results[0].score > results[1].score,
        "nearer vector score ({}) should exceed farther ({})",
        results[0].score,
        results[1].score
    );
}

#[tokio::test]
async fn memory_index_record_recall_then_recalled() {
    let test = setup().await;

    let from = test.store().create(None).await.unwrap();
    let to = test.store().create(None).await.unwrap();

    test.store()
        .record_recall(from.id, to.id, 0.9)
        .await
        .unwrap();

    let recalled = test.store().recalled(from.id).await.unwrap();
    assert_eq!(recalled.len(), 1);
    assert_eq!(recalled[0], to.id);
}

#[tokio::test]
async fn memory_index_search_honours_exclude() {
    let test = setup().await;

    let conv = test.store().create(None).await.unwrap();
    let embedding = unit_vec(0);
    test.store()
        .index_summary(conv.id, embedding.clone())
        .await
        .unwrap();

    // Exclude the only indexed conversation — result set must be empty.
    let results = MemoryIndex::search(test.store().as_ref(), embedding, 5, Some(conv.id))
        .await
        .unwrap();
    assert!(
        results.is_empty(),
        "excluded conversation must not appear in results"
    );
}

#[tokio::test]
async fn memory_index_record_recall_is_idempotent() {
    let test = setup().await;

    let from = test.store().create(None).await.unwrap();
    let to = test.store().create(None).await.unwrap();

    // Record the same edge twice with different scores.
    test.store()
        .record_recall(from.id, to.id, 0.7)
        .await
        .unwrap();
    test.store()
        .record_recall(from.id, to.id, 0.9)
        .await
        .unwrap();

    // Only one edge should exist between the pair.
    let recalled = test.store().recalled(from.id).await.unwrap();
    assert_eq!(
        recalled.len(),
        1,
        "idempotent call must not duplicate the edge"
    );
    assert_eq!(recalled[0], to.id);
}

// ── ToolIndex tests ─────────────────────────────────────────────────────────

/// A mock embedder that produces a deterministic unit vector per text input,
/// so tests can control similarity without a real embedding model.
///
/// The vector is 1024-dim with a single hot dimension chosen by hashing the
/// text. Two inputs that hash to the same dimension produce cosine = 1.0;
/// different dimensions produce cosine = 0.0. This mirrors the `unit_vec`
/// helper used by the MemoryIndex tests.
struct HashingMockEmbedder;

#[async_trait::async_trait]
impl Embedder for HashingMockEmbedder {
    async fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        text.hash(&mut hasher);
        let hot = (Hasher::finish(&hasher) as usize) % EMBED_DIMENSION;
        Ok(unit_vec(hot))
    }
}

fn hidden_registry_with(name: &str, description: &str) -> MockToolRegistry {
    MockToolRegistry::new().register_hidden(
        name,
        description,
        Policy::Auto,
        vec![ToolOutput {
            text: "ok".to_string(),
            is_error: false,
        }],
    )
}

#[tokio::test]
async fn tool_index_search_finds_indexed_hidden_tool() {
    let test = setup().await;
    let registry = hidden_registry_with("desktop_click", "Click an element via AX press.");
    let embedder = HashingMockEmbedder;

    test.store().index(&registry, &embedder).await.unwrap();

    // Search with the same text the index used — the embedder is deterministic,
    // so the query vector matches the stored vector exactly (cosine = 1.0).
    let query = embedder
        .embed("desktop_click\nClick an element via AX press.")
        .await
        .unwrap();
    let results = ToolIndex::search(test.store().as_ref(), query, 5)
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].name, "desktop_click");
    assert!((results[0].score - 1.0).abs() < 1e-4);
}

#[tokio::test]
async fn tool_index_search_returns_top_k_ordered_by_descending_score() {
    let test = setup().await;
    let registry = MockToolRegistry::new()
        .register_hidden("a", "alpha", Policy::Auto, vec![])
        .register_hidden("b", "beta", Policy::Auto, vec![])
        .register_hidden("c", "gamma", Policy::Auto, vec![]);
    let embedder = HashingMockEmbedder;

    test.store().index(&registry, &embedder).await.unwrap();

    // Query with tool "a"'s exact embedding text — "a" should rank first.
    let query = embedder.embed("a\nalpha").await.unwrap();
    let results = ToolIndex::search(test.store().as_ref(), query, 2)
        .await
        .unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].name, "a");
    assert!(results[0].score >= results[1].score);
}

#[tokio::test]
async fn tool_index_search_with_zero_k_returns_empty() {
    let test = setup().await;
    let registry = hidden_registry_with("solo", "the only hidden tool");
    let embedder = HashingMockEmbedder;

    test.store().index(&registry, &embedder).await.unwrap();

    let results = ToolIndex::search(test.store().as_ref(), vec![0.0; EMBED_DIMENSION], 0)
        .await
        .unwrap();
    assert!(results.is_empty());
}

#[tokio::test]
async fn tool_index_search_before_index_returns_empty() {
    let test = setup().await;
    let results = ToolIndex::search(test.store().as_ref(), vec![0.0; EMBED_DIMENSION], 5)
        .await
        .unwrap();
    assert!(results.is_empty());
}

#[tokio::test]
async fn tool_index_reindex_replaces_previous_entries() {
    let test = setup().await;
    let embedder = HashingMockEmbedder;

    let first = hidden_registry_with("old_tool", "the old hidden tool");
    test.store().index(&first, &embedder).await.unwrap();

    let second = hidden_registry_with("new_tool", "the new hidden tool");
    test.store().index(&second, &embedder).await.unwrap();

    let query = embedder
        .embed("new_tool\nthe new hidden tool")
        .await
        .unwrap();
    let results = ToolIndex::search(test.store().as_ref(), query, 10)
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].name, "new_tool");
}

#[tokio::test]
async fn tool_index_only_indexes_hidden_tools() {
    let test = setup().await;
    let registry = MockToolRegistry::new()
        .register_hidden("hidden_one", "A hidden tool.", Policy::Auto, vec![])
        .register("visible_one", "A visible tool.", Policy::Auto, vec![]);
    let embedder = HashingMockEmbedder;

    test.store().index(&registry, &embedder).await.unwrap();

    // A broad query should only ever return the hidden tool.
    let query = embedder.embed("hidden_one\nA hidden tool.").await.unwrap();
    let results = ToolIndex::search(test.store().as_ref(), query, 10)
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].name, "hidden_one");
}

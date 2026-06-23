use std::sync::Arc;

use brain::{
    conversation_store::ConversationStore,
    memory_index::MemoryIndex,
    models::{
        message::{Content, Message, MessageId, Role},
        summary::Summary,
    },
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

    let results = test.store().search(embedding, 5, None).await.unwrap();
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
    let results = test.store().search(unit_vec(0), 5, None).await.unwrap();
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
    let results = test
        .store()
        .search(embedding, 5, Some(conv.id))
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

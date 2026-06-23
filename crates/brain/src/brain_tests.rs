use std::sync::Arc;

use futures::{StreamExt, stream};

use crate::{
    Brain,
    conversation_store::ConversationStore,
    language_model::{AssistantEvent, Prompt, PromptMessage, ToolCall},
    memory_index::RelatedConversation,
    models::{
        conversation::{Conversation, ConversationId, ConversationItem},
        message::{Content, Message, MessageId, Role},
        summary::Summary,
    },
    testing::{
        conversation_store::InMemoryConversationStore, embedder::MockEmbedder,
        language_model::MockLanguageModel, memory_index::MockMemoryIndex,
        summarizer::MockSummarizer, tool_registry::MockToolRegistry,
        topic_detector::MockTopicDetector,
    },
    tool::Policy,
    topic_detector::TopicShift,
};

fn make_message(role: Role, text: &str) -> Message {
    Message {
        id: MessageId::new(),
        role,
        content: vec![Content::Text {
            text: text.to_string(),
        }],
        timestamp: jiff::Timestamp::now(),
    }
}

fn make_conversation() -> Conversation {
    let now = jiff::Timestamp::now();
    Conversation {
        id: ConversationId::new(),
        parent: None,
        items: Vec::new(),
        summary: None,
        created_at: now,
        updated_at: now,
        resolved_at: None,
    }
}

fn input_stream(text: &str) -> crate::InputStream {
    Box::pin(stream::iter(vec![Content::Text {
        text: text.to_string(),
    }]))
}

async fn drain(stream: crate::language_model::ContentStream) {
    let mut stream = stream;
    while let Some(chunk) = stream.next().await {
        chunk.unwrap();
    }
}

fn default_embedder() -> Arc<MockEmbedder> {
    Arc::new(MockEmbedder::default())
}

fn default_memory_index() -> Arc<MockMemoryIndex> {
    Arc::new(MockMemoryIndex::new())
}

fn default_tools() -> Arc<MockToolRegistry> {
    Arc::new(MockToolRegistry::new())
}

async fn setup(store: &InMemoryConversationStore, messages: Vec<Message>) -> ConversationId {
    let conv = make_conversation();
    let conv_id = conv.id;
    store.add_conversation(conv);
    for msg in messages {
        store.append_message(conv_id, msg).await.unwrap();
    }
    conv_id
}

/// The `recalled` field is system-prompt-only data. This test locks the
/// contract: the field exists on `Prompt`, carries whatever the orchestrator
/// places there, and is not secretly transformed.
#[test]
fn prompt_recalled_field_round_trips() {
    let entries = vec![
        "A prior thread about Rust ownership".to_string(),
        "An earlier discussion on async runtimes".to_string(),
    ];
    let prompt = Prompt {
        summary: Some("Prior context".to_string()),
        recalled: entries.clone(),
        messages: vec![PromptMessage {
            role: Role::User,
            content: vec![Content::Text {
                text: "hello".to_string(),
            }],
        }],
    };
    assert_eq!(prompt.recalled, entries);
    let empty = Prompt {
        summary: None,
        recalled: Vec::new(),
        messages: Vec::new(),
    };
    assert!(empty.recalled.is_empty());
}

#[tokio::test]
async fn say_calls_llm_with_user_message() {
    let store = Arc::new(InMemoryConversationStore::new());
    let _conv_id = setup(&store, vec![]).await;

    let llm = Arc::new(MockLanguageModel::new(1000));
    let brain = Brain::new(
        store,
        Arc::new(MockSummarizer::new("summary")),
        Arc::new(MockTopicDetector::no_shift()),
        llm.clone(),
        default_embedder(),
        default_memory_index(),
        default_tools(),
        100,
        0.8,
    );

    let stream = brain.say(input_stream("hello")).await.unwrap();
    drain(stream).await;

    let prompts = llm.prompts();
    let last = prompts.last().unwrap();
    let user_msg = last.messages.last().unwrap();
    assert!(matches!(user_msg.role, Role::User));
}

#[tokio::test]
async fn say_passes_summary_to_language_model() {
    let store = Arc::new(InMemoryConversationStore::new());
    let conv_id = setup(&store, vec![]).await;

    let summary = Summary {
        content: "Earlier we discussed Rust".to_string(),
        created_at: jiff::Timestamp::now(),
    };
    store.update_summary(conv_id, summary).await.unwrap();

    let llm = Arc::new(MockLanguageModel::new(1000));
    let brain = Brain::new(
        store,
        Arc::new(MockSummarizer::new("summary")),
        Arc::new(MockTopicDetector::no_shift()),
        llm.clone(),
        default_embedder(),
        default_memory_index(),
        default_tools(),
        100,
        0.8,
    );

    let stream = brain.say(input_stream("hello")).await.unwrap();
    drain(stream).await;

    let prompt = &llm.prompts()[0];
    assert_eq!(prompt.summary.as_deref(), Some("Earlier we discussed Rust"));
}

#[tokio::test]
async fn say_persists_assistant_response() {
    let store = Arc::new(InMemoryConversationStore::new());
    let conv_id = setup(&store, vec![]).await;

    let llm = Arc::new(MockLanguageModel::with_chunks(
        1000,
        vec!["world".to_string()],
    ));
    let brain = Brain::new(
        store.clone(),
        Arc::new(MockSummarizer::new("summary")),
        Arc::new(MockTopicDetector::no_shift()),
        llm,
        default_embedder(),
        default_memory_index(),
        default_tools(),
        100,
        0.8,
    );

    let mut stream = brain.say(input_stream("hello")).await.unwrap();
    while let Some(chunk) = stream.next().await {
        chunk.unwrap();
    }

    for _ in 0..10 {
        tokio::task::yield_now().await;
    }

    let messages = store.list_messages(conv_id).await.unwrap();
    let assistant_count = messages
        .iter()
        .filter(|m| matches!(m.role, Role::Assistant))
        .count();
    assert_eq!(assistant_count, 1);
}

#[tokio::test]
async fn topic_shift_triggers_summarisation() {
    let store = Arc::new(InMemoryConversationStore::new());
    let msg1 = make_message(Role::User, "hello");
    let msg2 = make_message(Role::Assistant, "hi");
    let msg3 = make_message(Role::User, "let's talk about Rust");
    let shift_at = msg3.id;
    let conv_id = setup(&store, vec![msg1, msg2, msg3]).await;

    let topic_detector = Arc::new(MockTopicDetector::with_shift(TopicShift {
        at_message_id: shift_at,
        new_topic: "Rust".to_string(),
    }));
    let summarizer = Arc::new(MockSummarizer::new("discussed greetings"));
    let llm = Arc::new(MockLanguageModel::new(1000));

    let brain = Brain::new(
        store.clone(),
        summarizer.clone(),
        topic_detector,
        llm,
        default_embedder(),
        default_memory_index(),
        default_tools(),
        100,
        0.8,
    );

    let _stream = brain.say(input_stream("tell me more")).await.unwrap();

    let calls = summarizer.calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].len(), 2);

    let new_current = store.current().await.unwrap();
    assert_ne!(new_current, conv_id);

    let old_conv = store.get_conversation(conv_id).unwrap();
    assert!(old_conv.summary.is_none());

    let new_conv = store.get_conversation(new_current).unwrap();
    assert_eq!(new_conv.parent, Some(conv_id));
    assert!(new_conv.summary.is_some());
}

#[tokio::test]
async fn drift_split_carries_no_overlap() {
    let store = Arc::new(InMemoryConversationStore::new());
    let msg1 = make_message(Role::User, "hello");
    let msg2 = make_message(Role::Assistant, "hi");
    let msg3 = make_message(Role::User, "let's talk about Rust");
    let shift_at = msg3.id;
    let _conv_id = setup(&store, vec![msg1, msg2, msg3]).await;

    let topic_detector = Arc::new(MockTopicDetector::with_shift(TopicShift {
        at_message_id: shift_at,
        new_topic: "Rust".to_string(),
    }));

    let brain = Brain::new(
        store.clone(),
        Arc::new(MockSummarizer::new("discussed greetings")),
        topic_detector,
        Arc::new(MockLanguageModel::new(1000)),
        default_embedder(),
        default_memory_index(),
        default_tools(),
        100,
        0.8,
    );

    let _stream = brain.say(input_stream("tell me more")).await.unwrap();

    let new_current = store.current().await.unwrap();
    let new_conv = store.get_conversation(new_current).unwrap();
    let referenced = new_conv
        .items
        .iter()
        .filter(|item| matches!(item, ConversationItem::ReferencedMessage(_)))
        .count();
    assert_eq!(
        referenced, 0,
        "drift child must carry no referenced overlap"
    );
}

#[tokio::test]
async fn low_capacity_triggers_reorganise() {
    let store = Arc::new(InMemoryConversationStore::new());
    let messages: Vec<Message> = (0..5)
        .map(|i| {
            make_message(
                if i % 2 == 0 {
                    Role::User
                } else {
                    Role::Assistant
                },
                &format!("msg {}", i),
            )
        })
        .collect();
    let conv_id = setup(&store, messages).await;

    let llm = Arc::new(MockLanguageModel::new(50));
    let brain = Brain::new(
        store.clone(),
        Arc::new(MockSummarizer::new("compressed")),
        Arc::new(MockTopicDetector::no_shift()),
        llm,
        default_embedder(),
        default_memory_index(),
        default_tools(),
        100,
        0.8,
    );

    let _stream = brain.say(input_stream("hello")).await.unwrap();

    let new_current = store.current().await.unwrap();
    assert_ne!(new_current, conv_id);

    let new_conv = store.get_conversation(new_current).unwrap();
    assert_eq!(new_conv.parent, Some(conv_id));
}

#[tokio::test]
async fn split_first_reference_is_a_user_message() {
    let store = Arc::new(InMemoryConversationStore::new());
    let messages = vec![
        make_message(Role::User, "u1"),
        make_message(Role::Assistant, "a1"),
        make_message(Role::User, "u2"),
        make_message(Role::Assistant, "a2"),
        make_message(Role::User, "u3"),
    ];
    let conv_id = setup(&store, messages).await;

    let llm = Arc::new(MockLanguageModel::new(50));
    let brain = Brain::new(
        store.clone(),
        Arc::new(MockSummarizer::new("compressed")),
        Arc::new(MockTopicDetector::no_shift()),
        llm,
        default_embedder(),
        default_memory_index(),
        default_tools(),
        100,
        0.8,
    );

    let _stream = brain.say(input_stream("hello")).await.unwrap();

    let new_current = store.current().await.unwrap();
    assert_ne!(new_current, conv_id);

    let new_conv = store.get_conversation(new_current).unwrap();
    let first_ref = new_conv
        .items
        .iter()
        .find_map(|item| match item {
            ConversationItem::ReferencedMessage(id) => Some(*id),
            ConversationItem::Message(_) => None,
        })
        .expect("expected at least one referenced message");
    let first_ref_message = store.get_message(first_ref).unwrap();
    assert!(matches!(first_ref_message.role, Role::User));
}

#[tokio::test]
async fn reorganise_only_summarises_owned_messages() {
    let store = Arc::new(InMemoryConversationStore::new());
    let owned1 = make_message(Role::User, "owned 1");
    let owned2 = make_message(Role::User, "owned 2");
    let owned3 = make_message(Role::User, "owned 3");
    let referenced = make_message(Role::User, "referenced");
    let conv_id = setup(&store, vec![owned1, owned2, owned3]).await;
    store.add_referenced_message(conv_id, referenced.clone());

    let summarizer = Arc::new(MockSummarizer::new("compressed"));
    let llm = Arc::new(MockLanguageModel::new(50));
    let brain = Brain::new(
        store.clone(),
        summarizer.clone(),
        Arc::new(MockTopicDetector::no_shift()),
        llm,
        default_embedder(),
        default_memory_index(),
        default_tools(),
        100,
        0.8,
    );

    let _stream = brain.say(input_stream("hello")).await.unwrap();

    let calls = summarizer.calls();
    assert!(!calls.is_empty());
    let last_call = &calls[calls.len() - 1];
    for msg in last_call {
        assert_ne!(msg.id, referenced.id);
    }
}

#[tokio::test]
async fn topic_shift_records_recall_and_indexes_child_summary() {
    let store = Arc::new(InMemoryConversationStore::new());
    let prior = make_message(Role::User, "we discussed the CLI");
    let shift = make_message(Role::User, "now let's talk about vector recall");
    let parent = setup(&store, vec![prior, shift.clone()]).await;

    let candidate = make_conversation();
    let candidate_id = candidate.id;
    store.add_conversation(candidate);
    store
        .update_summary(
            candidate_id,
            Summary {
                content: "Earlier vector recall notes".to_string(),
                created_at: jiff::Timestamp::now(),
            },
        )
        .await
        .unwrap();

    let topic_detector = Arc::new(MockTopicDetector::with_shift(TopicShift {
        at_message_id: shift.id,
        new_topic: "vector recall".to_string(),
    }));
    let summarizer = Arc::new(MockSummarizer::new("child summary"));
    let llm = Arc::new(MockLanguageModel::new(1000));
    let embedder = Arc::new(MockEmbedder::new(3));
    let memory_index = Arc::new(MockMemoryIndex::new());
    memory_index.enable_index_signal();
    let notify = memory_index.index_notify().unwrap();
    memory_index.push_search(vec![RelatedConversation {
        id: candidate_id,
        score: 0.91,
    }]);

    let brain = Brain::new(
        store.clone(),
        summarizer,
        topic_detector,
        llm,
        embedder.clone(),
        memory_index.clone(),
        default_tools(),
        100,
        0.8,
    );

    let stream = brain.say(input_stream("tell me more")).await.unwrap();
    drain(stream).await;
    notify.notified().await;

    let child = store.current().await.unwrap();
    assert_ne!(child, parent);

    let recalls = memory_index.recalls();
    assert_eq!(recalls.len(), 1);
    assert_eq!(recalls[0].from, child);
    assert_eq!(recalls[0].to, candidate_id);
    assert_eq!(recalls[0].score, 0.91);

    let search_calls = memory_index.search_calls();
    assert_eq!(search_calls.len(), 1);
    assert_eq!(search_calls[0].k, 1);
    assert_eq!(search_calls[0].exclude, Some(parent));

    let indexed = memory_index.indexed();
    assert_eq!(indexed.len(), 1);
    assert_eq!(indexed[0].id, child);
    assert_eq!(indexed[0].embedding, vec![0.0, 0.0, 0.0]);

    let embed_calls = embedder.calls();
    assert!(embed_calls.contains(&"vector recall".to_string()));
    assert!(embed_calls.contains(&"child summary".to_string()));
}

#[tokio::test]
async fn following_turn_resolves_recalled_summary_into_prompt_only() {
    let store = Arc::new(InMemoryConversationStore::new());
    let parent_message = make_message(Role::User, "old thread");
    let shift = make_message(Role::User, "new thread");
    let _parent = setup(&store, vec![parent_message, shift.clone()]).await;

    let recalled_source = make_conversation();
    let recalled_source_id = recalled_source.id;
    store.add_conversation(recalled_source);
    store
        .update_summary(
            recalled_source_id,
            Summary {
                content: "Recalled summary from another thread".to_string(),
                created_at: jiff::Timestamp::now(),
            },
        )
        .await
        .unwrap();

    let llm = Arc::new(MockLanguageModel::with_chunks(
        1000,
        vec!["assistant output".to_string()],
    ));
    let memory_index = Arc::new(MockMemoryIndex::new());
    memory_index.push_search(vec![RelatedConversation {
        id: recalled_source_id,
        score: 0.95,
    }]);

    let brain = Brain::new(
        store.clone(),
        Arc::new(MockSummarizer::new("child summary")),
        Arc::new(MockTopicDetector::with_shift(TopicShift {
            at_message_id: shift.id,
            new_topic: "old thread".to_string(),
        })),
        llm.clone(),
        default_embedder(),
        memory_index,
        default_tools(),
        100,
        0.8,
    );

    let first = brain.say(input_stream("split now")).await.unwrap();
    drain(first).await;
    for _ in 0..10 {
        tokio::task::yield_now().await;
    }

    let child = store.current().await.unwrap();
    let second = brain.say(input_stream("continue")).await.unwrap();
    drain(second).await;
    for _ in 0..10 {
        tokio::task::yield_now().await;
    }

    let prompts = llm.prompts();
    assert_eq!(
        prompts.last().unwrap().recalled,
        vec!["Recalled summary from another thread".to_string()]
    );

    let messages = store.list_messages(child).await.unwrap();
    assert!(messages.iter().all(|m| !matches!(m.role, Role::System)));
    let text = messages
        .iter()
        .flat_map(|m| &m.content)
        .map(|c| match c {
            Content::Text { text } => text.as_str(),
            Content::ToolCall { .. }
            | Content::ToolResult { .. }
            | Content::TemporalUpdate { .. } => "",
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert!(!text.contains("Recalled summary from another thread"));
    assert!(text.contains("assistant output"));
}

#[tokio::test]
async fn candidate_below_threshold_produces_no_recall_edge() {
    let store = Arc::new(InMemoryConversationStore::new());
    let prior = make_message(Role::User, "prior");
    let shift = make_message(Role::User, "shift");
    setup(&store, vec![prior, shift.clone()]).await;

    let candidate_id = ConversationId::new();
    let memory_index = Arc::new(MockMemoryIndex::new());
    memory_index.push_search(vec![RelatedConversation {
        id: candidate_id,
        score: 0.79,
    }]);

    let brain = Brain::new(
        store.clone(),
        Arc::new(MockSummarizer::new("summary")),
        Arc::new(MockTopicDetector::with_shift(TopicShift {
            at_message_id: shift.id,
            new_topic: "shift".to_string(),
        })),
        Arc::new(MockLanguageModel::new(1000)),
        default_embedder(),
        memory_index.clone(),
        default_tools(),
        100,
        0.8,
    );

    let stream = brain.say(input_stream("continue")).await.unwrap();
    drain(stream).await;

    assert!(memory_index.recalls().is_empty());
}

#[tokio::test]
async fn compaction_split_does_not_search_or_record_recall() {
    let store = Arc::new(InMemoryConversationStore::new());
    let messages = vec![
        make_message(Role::User, "u1"),
        make_message(Role::Assistant, "a1"),
    ];
    setup(&store, messages).await;

    let memory_index = Arc::new(MockMemoryIndex::new());
    let brain = Brain::new(
        store,
        Arc::new(MockSummarizer::new("summary")),
        Arc::new(MockTopicDetector::no_shift()),
        Arc::new(MockLanguageModel::new(1)),
        default_embedder(),
        memory_index.clone(),
        default_tools(),
        100,
        0.8,
    );

    let stream = brain.say(input_stream("overflow")).await.unwrap();
    drain(stream).await;

    assert!(memory_index.search_calls().is_empty());
    assert!(memory_index.recalls().is_empty());
}

/// An `Auto` tool call executes, its call+result pair is persisted adjacent in
/// the parent conversation, the loop continues, and only text reaches the
/// `say` stream (no `ToolCall` event leaks out).
#[tokio::test]
async fn auto_tool_executes_and_loop_continues() {
    let store = Arc::new(InMemoryConversationStore::new());
    let conv_id = setup(&store, vec![]).await;

    // Turn 1: tool call, then turn 2: plain text after the result.
    let llm = Arc::new(MockLanguageModel::with_turns(
        1000,
        vec![
            vec![AssistantEvent::ToolCall(ToolCall {
                id: "call-1".to_string(),
                name: "echo".to_string(),
                input: serde_json::json!({ "text": "hello" }),
            })],
            vec![AssistantEvent::Text("done".to_string())],
        ],
    ));

    let tools = Arc::new(MockToolRegistry::new().register(
        "echo",
        "Echoes text",
        Policy::Auto,
        vec![crate::tool::ToolOutput {
            text: "hello".to_string(),
            is_error: false,
        }],
    ));

    let brain = Brain::new(
        store.clone(),
        Arc::new(MockSummarizer::new("sum")),
        Arc::new(MockTopicDetector::no_shift()),
        llm.clone(),
        default_embedder(),
        default_memory_index(),
        tools.clone(),
        100,
        0.8,
    );

    let mut text_chunks = Vec::new();
    let mut stream = brain.say(input_stream("run echo")).await.unwrap();
    while let Some(chunk) = stream.next().await {
        text_chunks.push(chunk.unwrap());
    }

    assert_eq!(text_chunks, vec!["done".to_string()]);

    let execs = tools.executions();
    assert_eq!(execs.len(), 1);
    assert_eq!(execs[0].0, "echo");

    for _ in 0..20 {
        tokio::task::yield_now().await;
    }

    let messages = store.list_messages(conv_id).await.unwrap();
    let has_tool_call = messages.iter().any(|m| {
        m.content
            .iter()
            .any(|c| matches!(c, Content::ToolCall { .. }))
    });
    let has_tool_result = messages.iter().any(|m| {
        m.content
            .iter()
            .any(|c| matches!(c, Content::ToolResult { .. }))
    });
    let has_assistant_text = messages.iter().any(|m| {
        matches!(m.role, Role::Assistant)
            && m.content.iter().any(|c| matches!(c, Content::Text { .. }))
    });
    assert!(has_tool_call, "parent must contain ToolCall content");
    assert!(has_tool_result, "parent must contain ToolResult content");
    assert!(has_assistant_text, "parent must contain assistant text");

    assert_eq!(llm.prompts().len(), 2);
}

/// The `ToolCall` + `ToolResult` pair is always adjacent in the conversation.
/// No other message may appear between a ToolCall and its matching ToolResult.
#[tokio::test]
async fn tool_call_and_result_are_adjacent_in_conversation() {
    let store = Arc::new(InMemoryConversationStore::new());
    let conv_id = setup(&store, vec![]).await;

    let llm = Arc::new(MockLanguageModel::with_turns(
        1000,
        vec![
            vec![AssistantEvent::ToolCall(ToolCall {
                id: "call-adj".to_string(),
                name: "echo".to_string(),
                input: serde_json::json!({}),
            })],
            vec![AssistantEvent::Text("all done".to_string())],
        ],
    ));

    let tools = Arc::new(MockToolRegistry::new().register(
        "echo",
        "Echoes text",
        Policy::Auto,
        vec![crate::tool::ToolOutput {
            text: "ok".to_string(),
            is_error: false,
        }],
    ));

    let brain = Brain::new(
        store.clone(),
        Arc::new(MockSummarizer::new("sum")),
        Arc::new(MockTopicDetector::no_shift()),
        llm,
        default_embedder(),
        default_memory_index(),
        tools,
        100,
        0.8,
    );

    let stream = brain.say(input_stream("go")).await.unwrap();
    drain(stream).await;
    for _ in 0..20 {
        tokio::task::yield_now().await;
    }

    let messages = store.list_messages(conv_id).await.unwrap();
    let pos_call = messages.iter().position(|m| {
        m.content
            .iter()
            .any(|c| matches!(c, Content::ToolCall { .. }))
    });
    let pos_result = messages.iter().position(|m| {
        m.content
            .iter()
            .any(|c| matches!(c, Content::ToolResult { .. }))
    });
    let pos_call = pos_call.expect("ToolCall message must exist");
    let pos_result = pos_result.expect("ToolResult message must exist");
    assert_eq!(
        pos_result,
        pos_call + 1,
        "ToolResult must immediately follow ToolCall"
    );
}

/// A `Confirm` tool call opens a child conversation, streams the model's
/// question to the user, writes nothing to the parent, and sets current to
/// the child so the next `say` lands there.
#[tokio::test]
async fn confirm_tool_opens_resolution_thread_and_writes_nothing_to_parent() {
    let store = Arc::new(InMemoryConversationStore::new());
    let parent_id = setup(&store, vec![]).await;

    let llm = Arc::new(MockLanguageModel::with_turns(
        1000,
        vec![
            vec![AssistantEvent::ToolCall(ToolCall {
                id: "call-confirm".to_string(),
                name: "send_email".to_string(),
                input: serde_json::json!({ "to": "alice@example.com" }),
            })],
            // The question the model asks:
            vec![AssistantEvent::Text(
                "About to send email. Shall I proceed?".to_string(),
            )],
        ],
    ));

    let tools = Arc::new(MockToolRegistry::new().register(
        "send_email",
        "Sends an email",
        Policy::Confirm,
        vec![crate::tool::ToolOutput {
            text: "sent".to_string(),
            is_error: false,
        }],
    ));

    let brain = Brain::new(
        store.clone(),
        Arc::new(MockSummarizer::new("sum")),
        Arc::new(MockTopicDetector::no_shift()),
        llm,
        default_embedder(),
        default_memory_index(),
        tools.clone(),
        100,
        0.8,
    );

    let mut question_text = String::new();
    let mut stream = brain.say(input_stream("send email")).await.unwrap();
    while let Some(chunk) = stream.next().await {
        question_text.push_str(&chunk.unwrap());
    }
    for _ in 0..20 {
        tokio::task::yield_now().await;
    }

    assert!(
        question_text.contains("email") || question_text.contains("proceed"),
        "expected a confirmation question, got: {question_text:?}"
    );

    let current = store.current().await.unwrap();
    assert_ne!(
        current, parent_id,
        "current must be the confirmation thread"
    );
    let thread = store.get_conversation(current).unwrap();
    assert_eq!(thread.parent, Some(parent_id));

    let parent_messages = store.list_messages(parent_id).await.unwrap();
    for msg in &parent_messages {
        for content in &msg.content {
            assert!(
                !matches!(
                    content,
                    Content::ToolCall { .. } | Content::ToolResult { .. }
                ),
                "parent must not contain tool call/result content before resolution"
            );
        }
    }

    assert!(tools.executions().is_empty());
}

/// The resolving turn in a confirmation thread: the model emits the
/// `__resolve` sentinel with `action=execute`, the tool runs, the adjacent
/// pair is written into the parent, and current moves back to the parent.
#[tokio::test]
async fn confirm_tool_resolves_execute_writes_pair_to_parent() {
    let store = Arc::new(InMemoryConversationStore::new());
    let parent_id = setup(&store, vec![]).await;

    let llm = Arc::new(MockLanguageModel::with_turns(
        1000,
        vec![
            vec![AssistantEvent::ToolCall(ToolCall {
                id: "call-r".to_string(),
                name: "send_email".to_string(),
                input: serde_json::json!({ "to": "bob@example.com" }),
            })],
            vec![AssistantEvent::Text("Shall I send?".to_string())],
            vec![AssistantEvent::ToolCall(ToolCall {
                id: "resolve-1".to_string(),
                name: "__resolve".to_string(),
                input: serde_json::json!({ "action": "execute" }),
            })],
        ],
    ));

    let tools = Arc::new(MockToolRegistry::new().register(
        "send_email",
        "Sends an email",
        Policy::Confirm,
        vec![crate::tool::ToolOutput {
            text: "email sent!".to_string(),
            is_error: false,
        }],
    ));

    let brain = Brain::new(
        store.clone(),
        Arc::new(MockSummarizer::new("sum")),
        Arc::new(MockTopicDetector::no_shift()),
        llm,
        default_embedder(),
        default_memory_index(),
        tools.clone(),
        100,
        0.8,
    );

    let stream = brain.say(input_stream("email bob")).await.unwrap();
    drain(stream).await;
    for _ in 0..20 {
        tokio::task::yield_now().await;
    }

    let thread_id = store.current().await.unwrap();
    assert_ne!(thread_id, parent_id);

    let stream = brain.say(input_stream("yes, send it")).await.unwrap();
    drain(stream).await;
    for _ in 0..20 {
        tokio::task::yield_now().await;
    }

    let current = store.current().await.unwrap();
    assert_eq!(
        current, parent_id,
        "current must return to parent after resolve"
    );

    let execs = tools.executions();
    assert_eq!(execs.len(), 1);
    assert_eq!(execs[0].0, "send_email");

    let parent_messages = store.list_messages(parent_id).await.unwrap();
    let has_tool_call = parent_messages.iter().any(|m| {
        m.content
            .iter()
            .any(|c| matches!(c, Content::ToolCall { .. }))
    });
    let has_tool_result = parent_messages.iter().any(|m| {
        m.content
            .iter()
            .any(|c| matches!(c, Content::ToolResult { .. }))
    });
    assert!(
        has_tool_call,
        "parent must contain ToolCall after resolution"
    );
    assert!(
        has_tool_result,
        "parent must contain ToolResult after resolution"
    );

    let thread_messages = store.list_messages(thread_id).await.unwrap();
    let thread_has_tool_result = thread_messages.iter().any(|m| {
        m.content
            .iter()
            .any(|c| matches!(c, Content::ToolResult { .. }))
    });
    assert!(
        !thread_has_tool_result,
        "thread must NOT contain ToolResult — that belongs in the parent"
    );
}

/// A `cancel` via the `__resolve` sentinel writes a cancellation `ToolResult`
/// into the parent, runs no tool, and returns current to the parent.
#[tokio::test]
async fn confirm_tool_resolves_cancel_writes_cancellation_to_parent() {
    let store = Arc::new(InMemoryConversationStore::new());
    let parent_id = setup(&store, vec![]).await;

    let llm = Arc::new(MockLanguageModel::with_turns(
        1000,
        vec![
            vec![AssistantEvent::ToolCall(ToolCall {
                id: "call-c".to_string(),
                name: "send_email".to_string(),
                input: serde_json::json!({ "to": "carol@example.com" }),
            })],
            vec![AssistantEvent::Text("Shall I send?".to_string())],
            vec![AssistantEvent::ToolCall(ToolCall {
                id: "resolve-c".to_string(),
                name: "__resolve".to_string(),
                input: serde_json::json!({ "action": "cancel" }),
            })],
        ],
    ));

    let tools = Arc::new(MockToolRegistry::new().register(
        "send_email",
        "Sends an email",
        Policy::Confirm,
        vec![crate::tool::ToolOutput {
            text: "sent".to_string(),
            is_error: false,
        }],
    ));

    let brain = Brain::new(
        store.clone(),
        Arc::new(MockSummarizer::new("sum")),
        Arc::new(MockTopicDetector::no_shift()),
        llm,
        default_embedder(),
        default_memory_index(),
        tools.clone(),
        100,
        0.8,
    );

    let stream = brain.say(input_stream("email carol")).await.unwrap();
    drain(stream).await;
    for _ in 0..20 {
        tokio::task::yield_now().await;
    }

    let stream = brain
        .say(input_stream("actually, don't send"))
        .await
        .unwrap();
    drain(stream).await;
    for _ in 0..20 {
        tokio::task::yield_now().await;
    }

    assert_eq!(store.current().await.unwrap(), parent_id);

    assert!(tools.executions().is_empty(), "tool must not run on cancel");

    let parent_messages = store.list_messages(parent_id).await.unwrap();
    let cancellation = parent_messages.iter().find(|m| {
        m.content
            .iter()
            .any(|c| matches!(c, Content::ToolResult { .. }))
    });
    assert!(
        cancellation.is_some(),
        "parent must have a cancellation ToolResult"
    );
}

/// `list_messages` on the parent shows the adjacent tool pair and no
/// confirmation thread dialogue.
#[tokio::test]
async fn list_messages_on_parent_shows_pair_not_thread_dialogue() {
    let store = Arc::new(InMemoryConversationStore::new());
    let parent_id = setup(&store, vec![]).await;

    let llm = Arc::new(MockLanguageModel::with_turns(
        1000,
        vec![
            vec![AssistantEvent::ToolCall(ToolCall {
                id: "call-lt".to_string(),
                name: "send_email".to_string(),
                input: serde_json::json!({ "to": "dave@example.com" }),
            })],
            vec![AssistantEvent::Text("Shall I proceed?".to_string())],
            vec![AssistantEvent::ToolCall(ToolCall {
                id: "resolve-lt".to_string(),
                name: "__resolve".to_string(),
                input: serde_json::json!({ "action": "execute" }),
            })],
        ],
    ));

    let tools = Arc::new(MockToolRegistry::new().register(
        "send_email",
        "Sends an email",
        Policy::Confirm,
        vec![crate::tool::ToolOutput {
            text: "sent".to_string(),
            is_error: false,
        }],
    ));

    let brain = Brain::new(
        store.clone(),
        Arc::new(MockSummarizer::new("sum")),
        Arc::new(MockTopicDetector::no_shift()),
        llm,
        default_embedder(),
        default_memory_index(),
        tools,
        100,
        0.8,
    );

    let stream = brain.say(input_stream("email dave")).await.unwrap();
    drain(stream).await;
    for _ in 0..20 {
        tokio::task::yield_now().await;
    }
    let stream = brain.say(input_stream("yes")).await.unwrap();
    drain(stream).await;
    for _ in 0..20 {
        tokio::task::yield_now().await;
    }

    let parent_messages = brain.list_messages(parent_id).await.unwrap();

    let pair_present = parent_messages.iter().any(|m| {
        m.content
            .iter()
            .any(|c| matches!(c, Content::ToolCall { .. }))
    });
    assert!(pair_present, "parent must contain the ToolCall");

    let dialogue_in_parent = parent_messages.iter().any(|m| {
        m.content.iter().any(|c| {
            if let Content::Text { text } = c {
                text.contains("proceed") || text.contains("Shall")
            } else {
                false
            }
        })
    });
    assert!(
        !dialogue_in_parent,
        "parent must not contain thread dialogue"
    );
}

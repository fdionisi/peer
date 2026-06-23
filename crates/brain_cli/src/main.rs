use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use brain::conversation_store::ConversationStore;
use brain::models::message::{Content, Role};
use brain::{
    Brain, InputStream,
    tool::{EmptyToolRegistry, StaticToolRegistry},
};
use brain_exa::{ExaConfig, ExaWebSearch};
use brain_mistralai::{MistralClient, MistralConfig};
use brain_prompts_embedded::EmbeddedPromptRegistry;
use brain_surrealdb::{SurrealDbClient, default_migrations_dir};
use futures::StreamExt;
use ratatui::{
    DefaultTerminal,
    crossterm::event::{self, Event, KeyCode},
};
use surrealdb::Surreal;
use surrealdb::engine::any::Any;
use tokio::sync::mpsc;

mod ui;

const REDRAW_INTERVAL: Duration = Duration::from_micros(62_500);
const DEFAULT_RECALL_THRESHOLD: f32 = 0.8;

#[derive(Clone)]
struct Config {
    mistral_api_key: String,
    mistral_model: String,
    mistral_embed_model: String,
    recall_threshold: f32,
    surrealdb_url: String,
    surrealdb_namespace: String,
    surrealdb_database: String,
    surrealdb_user: String,
    surrealdb_password: String,
    exa_api_key: Option<String>,
}

impl Config {
    fn from_env() -> Result<Self> {
        Ok(Self {
            mistral_api_key: std::env::var("MISTRAL_API_KEY")
                .context("MISTRAL_API_KEY must be set")?,
            mistral_model: std::env::var("MISTRAL_MODEL")
                .unwrap_or_else(|_| "mistral-small-latest".to_string()),
            mistral_embed_model: std::env::var("MISTRAL_EMBED_MODEL")
                .unwrap_or_else(|_| "mistral-embed".to_string()),
            recall_threshold: match std::env::var("RECALL_THRESHOLD") {
                Ok(raw) => raw
                    .parse::<f32>()
                    .context("RECALL_THRESHOLD must be a float")?,
                Err(_) => DEFAULT_RECALL_THRESHOLD,
            },
            surrealdb_url: std::env::var("SURREALDB_URL")
                .unwrap_or_else(|_| "ws://127.0.0.1:8000".to_string()),
            surrealdb_namespace: std::env::var("SURREALDB_NAMESPACE")
                .unwrap_or_else(|_| "brain".to_string()),
            surrealdb_database: std::env::var("SURREALDB_DATABASE")
                .unwrap_or_else(|_| "brain".to_string()),
            surrealdb_user: std::env::var("SURREALDB_USER").unwrap_or_else(|_| "root".to_string()),
            surrealdb_password: std::env::var("SURREALDB_PASSWORD")
                .unwrap_or_else(|_| "root".to_string()),
            exa_api_key: std::env::var("EXA_API_KEY").ok(),
        })
    }
}

#[derive(Clone)]
struct DisplayMessage {
    role: Role,
    content: String,
}

struct App {
    messages: Vec<DisplayMessage>,
    input: String,
    streaming: Option<String>,
    status: String,
    scroll: u16,
}

impl App {
    fn new() -> Self {
        Self {
            messages: Vec::new(),
            input: String::new(),
            streaming: None,
            status: "Ready".to_string(),
            scroll: 0,
        }
    }
}

enum AppEvent {
    UserMessage(String),
    Chunk(String),
    Done(String),
    Error(String),
}

async fn run(
    app: Arc<tokio::sync::Mutex<App>>,
    brain: Arc<Brain>,
    terminal: &mut DefaultTerminal,
) -> Result<()> {
    {
        let mut app = app.lock().await;
        let conv_id = brain.current_conversation().await?;
        let conv = brain
            .get_conversation(conv_id)
            .await?
            .context("conversation not found")?;
        let messages = brain.list_messages(conv_id).await?;

        app.messages = messages
            .into_iter()
            .filter_map(|m| {
                let content = extract_text(&m.content);
                if content.is_empty() {
                    None
                } else {
                    Some(DisplayMessage {
                        role: m.role,
                        content,
                    })
                }
            })
            .collect();

        app.status = match conv.summary {
            Some(_) => format!("{:?} with summary", conv_id),
            None => format!("{:?}", conv_id),
        };
    }

    let (tx, mut rx) = mpsc::unbounded_channel::<AppEvent>();
    let mut last_draw = Instant::now();

    loop {
        let now = Instant::now();
        if now.duration_since(last_draw) >= REDRAW_INTERVAL {
            {
                let app = app.lock().await;
                terminal.draw(|frame| ui::render(frame, &app))?;
            }
            last_draw = now;
        }

        while let Ok(event) = rx.try_recv() {
            let mut app = app.lock().await;
            match event {
                AppEvent::UserMessage(text) => {
                    app.messages.push(DisplayMessage {
                        role: Role::User,
                        content: text,
                    });
                    app.streaming = Some(String::new());
                    app.status = "Thinking...".to_string();
                    app.scroll = u16::MAX;
                }
                AppEvent::Chunk(text) => {
                    if let Some(ref mut s) = app.streaming {
                        s.push_str(&text);
                    }
                    app.scroll = u16::MAX;
                }
                AppEvent::Done(content) => {
                    app.messages.push(DisplayMessage {
                        role: Role::Assistant,
                        content,
                    });
                    app.streaming = None;
                    app.status = "Ready".to_string();
                    app.scroll = u16::MAX;
                }
                AppEvent::Error(e) => {
                    app.status = format!("Error: {e}");
                    app.streaming = None;
                }
            }
        }

        if event::poll(Duration::from_millis(16))? {
            if let Event::Key(key) = event::read()? {
                let mut app = app.lock().await;
                match key.code {
                    KeyCode::Char(c) => {
                        if key.modifiers.contains(event::KeyModifiers::CONTROL) && c == 'c' {
                            return Ok(());
                        }
                        app.input.push(c);
                    }
                    KeyCode::Backspace => {
                        app.input.pop();
                    }
                    KeyCode::Enter => {
                        let text = app.input.trim().to_string();
                        app.input.clear();
                        if !text.is_empty() {
                            let tx = tx.clone();
                            let brain = brain.clone();
                            tokio::spawn(async move {
                                send_message(brain, text, tx).await;
                            });
                        }
                    }
                    KeyCode::Esc => {
                        return Ok(());
                    }
                    KeyCode::PageUp => {
                        app.scroll = app.scroll.saturating_sub(10);
                    }
                    KeyCode::PageDown => {
                        app.scroll = app.scroll.saturating_add(10);
                    }
                    KeyCode::Up => {
                        app.scroll = app.scroll.saturating_sub(1);
                    }
                    KeyCode::Down => {
                        app.scroll = app.scroll.saturating_add(1);
                    }
                    KeyCode::End => {
                        app.scroll = u16::MAX;
                    }
                    _ => {}
                }
            }
        }
    }
}

async fn send_message(brain: Arc<Brain>, text: String, tx: mpsc::UnboundedSender<AppEvent>) {
    let _ = tx.send(AppEvent::UserMessage(text.clone()));

    let stream: InputStream = Box::pin(futures::stream::iter(vec![Content::Text {
        text: text.clone(),
    }]));

    let response_stream = match brain.say(stream).await {
        Ok(s) => s,
        Err(e) => {
            let _ = tx.send(AppEvent::Error(e.to_string()));
            return;
        }
    };

    let mut accumulated = String::new();
    let mut stream = response_stream;
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(text) => {
                accumulated.push_str(&text);
                let _ = tx.send(AppEvent::Chunk(text.to_string()));
            }
            Err(e) => {
                let _ = tx.send(AppEvent::Error(e.to_string()));
                return;
            }
        }
    }

    let _ = tx.send(AppEvent::Done(accumulated));
}

fn extract_text(content: &[Content]) -> String {
    content
        .iter()
        .filter_map(|c| match c {
            Content::Text { text } => Some(text.as_str()),
            Content::ToolCall { .. }
            | Content::ToolResult { .. }
            | Content::TemporalUpdate { .. } => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

#[tokio::main]
async fn main() -> Result<()> {
    let log_path = std::env::var("BRAIN_LOG").unwrap_or_else(|_| "brain.log".to_string());

    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .expect("failed to open log file");

    let (non_blocking, _guard) = tracing_appender::non_blocking(file);

    tracing_subscriber::fmt()
        .with_writer(non_blocking)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                tracing_subscriber::EnvFilter::new(
                    "brain=trace,brain_exa=trace,brain_mistralai=trace,brain_surrealdb=trace",
                )
            }),
        )
        .with_ansi(false)
        .init();

    let config = Config::from_env()?;

    let db: Surreal<Any> = Surreal::init();
    db.connect(&config.surrealdb_url).await?;
    db.signin(surrealdb::opt::auth::Root {
        username: config.surrealdb_user.clone(),
        password: config.surrealdb_password.clone(),
    })
    .await?;
    db.use_ns(&config.surrealdb_namespace)
        .use_db(&config.surrealdb_database)
        .await?;

    let store = Arc::new(SurrealDbClient::new(db));
    store.migrate(default_migrations_dir()).await?;

    let _current = match store.current().await {
        Ok(id) => id,
        Err(_) => {
            let conv = store.create(None).await?;
            store.set_current(conv.id.clone()).await?;
            conv.id
        }
    };

    let mistral_config = MistralConfig::new(&config.mistral_api_key, &config.mistral_model)
        .with_embed_model(&config.mistral_embed_model);
    let prompts: Arc<dyn brain::prompts::PromptRegistry> = Arc::new(EmbeddedPromptRegistry::new());
    let client = Arc::new(MistralClient::new(mistral_config, prompts)?);

    let tool_registry: Arc<dyn brain::tool::ToolRegistry> = match config.exa_api_key {
        Some(key) => Arc::new(StaticToolRegistry::new(vec![Box::new(ExaWebSearch::new(
            ExaConfig::new(key),
        )?)])),
        None => {
            tracing::warn!("EXA_API_KEY not set — starting without web search tool");
            Arc::new(EmptyToolRegistry)
        }
    };

    let brain = Arc::new(Brain::new(
        store.clone(),
        client.clone(),
        client.clone(),
        client.clone(),
        client,
        store,
        tool_registry,
        4096,
        config.recall_threshold,
    ));

    let app = Arc::new(tokio::sync::Mutex::new(App::new()));
    let mut terminal = ratatui::init();
    let result = run(app, brain, &mut terminal).await;
    ratatui::restore();
    result
}

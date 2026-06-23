pub mod client;

pub use client::MistralClient;

#[derive(Clone)]
pub struct MistralConfig {
    pub api_key: String,
    pub base_url: String,
    pub chat_model: String,
    pub summarizer_model: Option<String>,
    pub topic_detector_model: Option<String>,
    pub embed_model: Option<String>,
}

impl MistralConfig {
    pub fn new(api_key: impl Into<String>, chat_model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.mistral.ai/v1".to_string(),
            chat_model: chat_model.into(),
            summarizer_model: None,
            topic_detector_model: None,
            embed_model: None,
        }
    }

    pub fn with_summarizer_model(mut self, model: impl Into<String>) -> Self {
        self.summarizer_model = Some(model.into());
        self
    }

    pub fn with_topic_detector_model(mut self, model: impl Into<String>) -> Self {
        self.topic_detector_model = Some(model.into());
        self
    }

    pub fn with_embed_model(mut self, model: impl Into<String>) -> Self {
        self.embed_model = Some(model.into());
        self
    }
}

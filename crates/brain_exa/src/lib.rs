pub mod web_search;

pub use web_search::ExaWebSearch;

#[derive(Clone)]
pub struct ExaConfig {
    pub api_key: String,
    pub base_url: String,
}

impl ExaConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.exa.ai".to_string(),
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }
}

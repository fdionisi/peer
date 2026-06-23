use std::collections::HashMap;

use async_trait::async_trait;
use brain::prompts::{PromptRegistry, PromptTemplate};

pub struct EmbeddedPromptRegistry {
    templates: HashMap<&'static str, PromptTemplate>,
}

impl EmbeddedPromptRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            templates: HashMap::new(),
        };
        registry.register_all();
        registry
    }

    fn register_all(&mut self) {
        self.register(include_str!("../templates/system.tmpl"));
        self.register(include_str!("../templates/summarize_base.tmpl"));
        self.register(include_str!("../templates/summarize_topic_shift.tmpl"));
        self.register(include_str!("../templates/summarize_compaction.tmpl"));
        self.register(include_str!("../templates/detect_topic_shift.tmpl"));
    }

    fn register(&mut self, template: &'static str) {
        let name = template
            .lines()
            .next()
            .unwrap_or("")
            .trim_start_matches('#')
            .trim();
        self.templates
            .insert(name, PromptTemplate { name, template });
    }
}

impl Default for EmbeddedPromptRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl PromptRegistry for EmbeddedPromptRegistry {
    fn get(&self, name: &str) -> Option<&PromptTemplate> {
        self.templates.get(name)
    }

    fn names(&self) -> Vec<&str> {
        self.templates.keys().copied().collect()
    }
}

use std::collections::HashMap;

use async_trait::async_trait;

use crate::prompts::{PromptRegistry, PromptTemplate};

pub struct MockPromptRegistry {
    templates: HashMap<String, PromptTemplate>,
}

impl MockPromptRegistry {
    pub fn new() -> Self {
        Self {
            templates: HashMap::new(),
        }
    }

    pub fn with_template(mut self, name: &str, template: &str) -> Self {
        let name_static = Box::leak(name.to_string().into_boxed_str());
        let template_static = Box::leak(template.to_string().into_boxed_str());
        self.templates.insert(
            name.to_string(),
            PromptTemplate {
                name: name_static,
                template: template_static,
            },
        );
        self
    }
}

impl Default for MockPromptRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl PromptRegistry for MockPromptRegistry {
    fn get(&self, name: &str) -> Option<&PromptTemplate> {
        self.templates.get(name)
    }

    fn names(&self) -> Vec<&str> {
        self.templates.keys().map(|s| s.as_str()).collect()
    }
}

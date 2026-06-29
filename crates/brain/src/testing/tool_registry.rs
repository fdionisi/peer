use std::collections::HashMap;
use std::sync::Mutex;

use anyhow::Result;
use async_trait::async_trait;

use crate::tool::{Policy, ToolDefinition, ToolOutput, ToolRegistry, Visibility};

/// A scripted tool registry for tests. Callers register named tools with their
/// policy and a canned `ToolOutput`; the registry replays those outputs in
/// order on successive `execute` calls for each tool name.
pub struct MockToolRegistry {
    definitions: Vec<ToolDefinition>,
    policies: HashMap<String, Policy>,
    visibilities: HashMap<String, Visibility>,
    outputs: Mutex<HashMap<String, Vec<ToolOutput>>>,
    /// Records every `(name, input)` pair received by `execute`.
    pub executions: Mutex<Vec<(String, serde_json::Value)>>,
}

impl MockToolRegistry {
    pub fn new() -> Self {
        Self {
            definitions: Vec::new(),
            policies: HashMap::new(),
            visibilities: HashMap::new(),
            outputs: Mutex::new(HashMap::new()),
            executions: Mutex::new(Vec::new()),
        }
    }

    /// Register a tool with the given policy and a queue of canned outputs.
    /// Each `execute` call pops the first output; once exhausted, subsequent
    /// calls return an error.
    ///
    /// Registered tools are [`Visibility::Visible`] by default so existing
    /// tests keep seeing their tools in the default list; use
    /// [`MockToolRegistry::register_hidden`] for hidden tools.
    pub fn register(
        mut self,
        name: &str,
        description: &str,
        policy: Policy,
        outputs: Vec<ToolOutput>,
    ) -> Self {
        self.push_definition(name, description);
        self.policies.insert(name.to_string(), policy);
        self.visibilities
            .insert(name.to_string(), Visibility::Visible);
        self.outputs
            .lock()
            .unwrap()
            .insert(name.to_string(), outputs);
        self
    }

    /// Register a tool that is omitted from the default tool list.
    pub fn register_hidden(
        mut self,
        name: &str,
        description: &str,
        policy: Policy,
        outputs: Vec<ToolOutput>,
    ) -> Self {
        self.push_definition(name, description);
        self.policies.insert(name.to_string(), policy);
        self.visibilities
            .insert(name.to_string(), Visibility::Hidden);
        self.outputs
            .lock()
            .unwrap()
            .insert(name.to_string(), outputs);
        self
    }

    pub fn register_with_visibility(
        mut self,
        name: &str,
        description: &str,
        policy: Policy,
        visibility: Visibility,
        outputs: Vec<ToolOutput>,
    ) -> Self {
        self.push_definition(name, description);
        self.policies.insert(name.to_string(), policy);
        self.visibilities.insert(name.to_string(), visibility);
        self.outputs
            .lock()
            .unwrap()
            .insert(name.to_string(), outputs);
        self
    }

    fn push_definition(&mut self, name: &str, description: &str) {
        self.definitions.push(ToolDefinition {
            name: name.to_string(),
            description: description.to_string(),
            input_schema: serde_json::json!({ "type": "object" }),
        });
    }

    pub fn executions(&self) -> Vec<(String, serde_json::Value)> {
        self.executions.lock().unwrap().clone()
    }
}

impl Default for MockToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolRegistry for MockToolRegistry {
    fn names(&self) -> Vec<String> {
        self.definitions.iter().map(|d| d.name.clone()).collect()
    }

    fn definitions(&self) -> Vec<ToolDefinition> {
        self.definitions.clone()
    }

    fn policy(&self, name: &str) -> Option<Policy> {
        self.policies.get(name).copied()
    }

    fn visibility(&self, name: &str) -> Option<Visibility> {
        self.visibilities.get(name).copied()
    }

    async fn execute(&self, name: &str, input: serde_json::Value) -> Result<ToolOutput> {
        self.executions
            .lock()
            .unwrap()
            .push((name.to_string(), input));
        let mut outputs = self.outputs.lock().unwrap();
        let queue = outputs
            .get_mut(name)
            .ok_or_else(|| anyhow::anyhow!("no tool registered: {name}"))?;
        if queue.is_empty() {
            return Err(anyhow::anyhow!("no more canned outputs for tool: {name}"));
        }
        Ok(queue.remove(0))
    }
}

use anyhow::Result;
use async_trait::async_trait;

/// How a tool's resolution thread is run.
///
/// `Auto` tools resolve in an empty thread — zero turns, immediate execution.
/// `Confirm` tools resolve in a thread that carries a dialogue: the model asks
/// its question, the user answers, the model interprets the answer, and the
/// thread resolves to a decision (execute, execute-with-changes, or cancel).
///
/// The policy is a property of the tool itself because the risk profile is
/// intrinsic to what the tool does — web search is read-only, sending an email
/// is mutating, and that distinction belongs with the tool definition, not in
/// a separate allow-list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    Auto,
    Confirm,
}

/// The result of executing a tool.
///
/// `is_error` feeds `Content::ToolResult.is_error` — it tells the model whether
/// the tool succeeded or failed, so it can reason about the outcome and decide
/// what to do next.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolOutput {
    pub text: String,
    pub is_error: bool,
}

/// The metadata the language model sees for a tool — the MCP-aligned shape
/// passed to the model so it knows what tools are available and how to call
/// them.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// A self-describing capability the model can invoke.
///
/// The trait is the interface. Implementations carry no knowledge of HTTP,
/// providers, or rate limits — those live in the implementation.
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn input_schema(&self) -> serde_json::Value;
    fn policy(&self) -> Policy {
        Policy::Auto
    }
    async fn execute(&self, input: serde_json::Value) -> Result<ToolOutput>;

    /// Assemble the three metadata methods into a `ToolDefinition` for the
    /// language model. Provided so adapters and the registry never rebuild it
    /// by hand.
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: self.input_schema(),
        }
    }
}

/// Looks up tools by name. A static list of tools at startup is enough for now;
/// the registry is the boundary that lets us add dynamic tools (per-conversation
/// tools, user-installed tools, MCP-served tools) later without touching the
/// orchestrator.
#[async_trait]
pub trait ToolRegistry: Send + Sync {
    fn names(&self) -> Vec<String>;
    fn definitions(&self) -> Vec<ToolDefinition>;
    /// The orchestrator needs the policy without executing — to decide whether
    /// to open a confirmation thread or just run the tool.
    fn policy(&self, name: &str) -> Option<Policy>;
    async fn execute(&self, name: &str, input: serde_json::Value) -> Result<ToolOutput>;
}

/// A registry backed by a fixed list of [`Tool`] trait objects.
///
/// Construct it with [`StaticToolRegistry::new`], then pass it into [`Brain::new`].
/// The list is immutable after construction.
pub struct StaticToolRegistry {
    tools: Vec<Box<dyn Tool>>,
}

impl StaticToolRegistry {
    pub fn new(tools: Vec<Box<dyn Tool>>) -> Self {
        Self { tools }
    }
}

#[async_trait]
impl ToolRegistry for StaticToolRegistry {
    fn names(&self) -> Vec<String> {
        self.tools.iter().map(|t| t.name().to_string()).collect()
    }

    fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools.iter().map(|t| t.definition()).collect()
    }

    fn policy(&self, name: &str) -> Option<Policy> {
        self.tools
            .iter()
            .find(|t| t.name() == name)
            .map(|t| t.policy())
    }

    async fn execute(&self, name: &str, input: serde_json::Value) -> Result<ToolOutput> {
        let tool = self
            .tools
            .iter()
            .find(|t| t.name() == name)
            .ok_or_else(|| anyhow::anyhow!("no tool registered: {name}"))?;
        tool.execute(input).await
    }
}

/// A no-op registry with no tools registered. Useful as a default when
/// building a `Brain` without tool support.
pub struct EmptyToolRegistry;

#[async_trait]
impl ToolRegistry for EmptyToolRegistry {
    fn names(&self) -> Vec<String> {
        Vec::new()
    }

    fn definitions(&self) -> Vec<ToolDefinition> {
        Vec::new()
    }

    fn policy(&self, _name: &str) -> Option<Policy> {
        None
    }

    async fn execute(&self, name: &str, _input: serde_json::Value) -> Result<ToolOutput> {
        Err(anyhow::anyhow!(
            "no tools registered; cannot execute: {name}"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EchoTool;

    #[async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }

        fn description(&self) -> &str {
            "Echoes the input back as text."
        }

        fn input_schema(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "text": { "type": "string" }
                }
            })
        }

        async fn execute(&self, input: serde_json::Value) -> Result<ToolOutput> {
            Ok(ToolOutput {
                text: input.to_string(),
                is_error: false,
            })
        }
    }

    #[test]
    fn definition_assembles_metadata() {
        let tool = EchoTool;
        let def = tool.definition();
        assert_eq!(def.name, tool.name());
        assert_eq!(def.description, tool.description());
        assert_eq!(def.input_schema, tool.input_schema());
    }

    #[test]
    fn default_policy_is_auto() {
        let tool = EchoTool;
        assert_eq!(tool.policy(), Policy::Auto);
    }
}

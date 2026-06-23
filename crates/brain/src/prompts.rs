use std::collections::BTreeMap;

use async_trait::async_trait;

#[derive(Debug, Clone)]
pub struct PromptTemplate {
    pub name: &'static str,
    pub template: &'static str,
}

#[async_trait]
pub trait PromptRegistry: Send + Sync {
    fn get(&self, name: &str) -> Option<&PromptTemplate>;
    fn names(&self) -> Vec<&str>;
}

/// Renders a prompt template with the MiniJinja engine.
///
/// The first line is a `#name` metadata header — it identifies the template
/// within a [`PromptRegistry`] and is stripped from the output. The remaining
/// body is a full MiniJinja template, so it supports variables (`{{ x }}`),
/// conditionals (`{% if x %}...{% endif %}`), loops, and filters, evaluated
/// against `vars`.
///
/// Variables that are absent from `vars` are treated as undefined: they render
/// as empty and are falsy, so optional sections degrade cleanly without the
/// caller having to pass placeholder values.
pub fn render(template: &PromptTemplate, vars: &[(&str, &str)]) -> anyhow::Result<String> {
    let body = strip_header(template.template);
    let context: BTreeMap<&str, &str> = vars.iter().copied().collect();

    let env = minijinja::Environment::new();
    let rendered = env
        .render_str(body, context)
        .map_err(|e| anyhow::anyhow!("failed to render prompt '{}': {e}", template.name))?;

    Ok(rendered.trim().to_string())
}

/// Strips the leading `#name` metadata line, leaving the renderable body.
fn strip_header(template: &str) -> &str {
    match template.split_once('\n') {
        Some((first, rest)) if first.trim_start().starts_with('#') => rest,
        _ => template,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_treats_missing_conditional_vars_as_absent() {
        let template = PromptTemplate {
            name: "system",
            template: "#system\nBase{% if recalled %}\n<recalled>{{ recalled }}</recalled>{% endif %}",
        };

        let rendered = render(&template, &[]).unwrap();

        assert_eq!(rendered, "Base");
    }

    #[test]
    fn render_includes_supplied_conditional_vars() {
        let template = PromptTemplate {
            name: "system",
            template: "#system\nBase{% if recalled %}\n<recalled>{{ recalled }}</recalled>{% endif %}",
        };

        let rendered = render(&template, &[("recalled", "Prior summary")]).unwrap();

        assert_eq!(rendered, "Base\n<recalled>Prior summary</recalled>");
    }
}

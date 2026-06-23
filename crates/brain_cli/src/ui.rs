use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Paragraph, Wrap},
};

use crate::App;

pub fn render(frame: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(1),
            Constraint::Length(3),
            Constraint::Length(1),
        ])
        .split(frame.area());

    let conversation_width = chunks[0].width.saturating_sub(2) as usize;
    let conversation_height = chunks[0].height.saturating_sub(2) as usize;

    let text = build_text(app);
    let total_lines = count_wrapped_lines(&text, conversation_width);
    let max_scroll = total_lines.saturating_sub(conversation_height) as u16;
    let scroll = app.scroll.min(max_scroll);

    let conversation = Paragraph::new(text)
        .block(Block::default().borders(Borders::ALL).title("Conversation"))
        .wrap(Wrap { trim: false })
        .scroll((scroll, 0));
    frame.render_widget(conversation, chunks[0]);

    let input = Paragraph::new(app.input.as_str())
        .block(Block::default().borders(Borders::ALL).title("Input"))
        .wrap(Wrap { trim: false });
    frame.render_widget(input, chunks[1]);

    let status = Paragraph::new(app.status.as_str()).style(Style::default().fg(Color::DarkGray));
    frame.render_widget(status, chunks[2]);
}

fn build_text(app: &App) -> Text<'_> {
    let mut text = Text::default();

    for m in &app.messages {
        let (prefix, color) = match m.role {
            brain::models::message::Role::User => ("You: ", Color::Cyan),
            brain::models::message::Role::Assistant => ("Brain: ", Color::Green),
            brain::models::message::Role::System => ("System: ", Color::Yellow),
        };

        text.push_line(Line::from(Span::styled(
            prefix,
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        )));

        if matches!(m.role, brain::models::message::Role::Assistant) {
            text.extend(tui_markdown::from_str(&m.content));
        } else {
            text.push_line(Line::from(m.content.clone()));
        }

        text.push_line(Line::from(""));
    }

    if let Some(streaming) = &app.streaming {
        text.push_line(Line::from(Span::styled(
            "Brain: ",
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        )));
        text.extend(tui_markdown::from_str(streaming));
    }

    text
}

fn count_wrapped_lines(text: &Text, width: usize) -> usize {
    if width == 0 {
        return text.lines.len();
    }
    let mut total = 0;
    for line in &text.lines {
        let line_width: usize = line.spans.iter().map(|s| s.content.chars().count()).sum();
        if line_width == 0 {
            total += 1;
        } else {
            total += line_width.div_ceil(width);
        }
    }
    total
}

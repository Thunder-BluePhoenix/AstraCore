/// Terminal TUI backend — interactive dashboard using ratatui + crossterm.
///
/// Layout (two-column):
/// ```text
/// ┌─────────────────────┬──────────────────────────────────────────────┐
/// │  ⚛  AstraCore Dashboard  (press q or Esc to quit)                  │
/// ├─────────────────────┼──────────────────────────────────────────────┤
/// │  Circuit Metrics    │  Probability Distribution                     │
/// │  Execution Stats    ├──────────────────────────────────────────────┤
/// │                     │  Gate Histogram                               │
/// │                     ├──────────────────────────────────────────────┤
/// │                     │  Qubit Utilization                            │
/// └─────────────────────┴──────────────────────────────────────────────┘
/// ```
///
/// Key bindings:
///   `q` / `Esc`  — quit
use crossterm::{
    event::{self, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Bar, BarChart, BarGroup, Block, Borders, Paragraph},
    Frame, Terminal,
};
use std::io::{self, Stdout};

use crate::dashboard::DashboardData;

// ── Public API ────────────────────────────────────────────────────────────

/// Launch the interactive TUI dashboard.
///
/// Draws the dashboard and blocks until the user presses `q` or `Esc`.
/// Restores the terminal on exit (including on panics via the drop impl).
pub fn run_tui(data: &DashboardData) -> io::Result<()> {
    let mut terminal = setup_terminal()?;
    let result = tui_loop(&mut terminal, data);
    restore_terminal(terminal)?;
    result
}

// ── Terminal lifecycle ─────────────────────────────────────────────────────

fn setup_terminal() -> io::Result<Terminal<CrosstermBackend<Stdout>>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    Terminal::new(CrosstermBackend::new(stdout))
}

fn restore_terminal(mut terminal: Terminal<CrosstermBackend<Stdout>>) -> io::Result<()> {
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()
}

// ── Main event loop ───────────────────────────────────────────────────────

fn tui_loop(terminal: &mut Terminal<CrosstermBackend<Stdout>>, data: &DashboardData) -> io::Result<()> {
    loop {
        terminal.draw(|frame| draw_ui(frame, data))?;

        if event::poll(std::time::Duration::from_millis(250))? {
            if let Event::Key(key) = event::read()? {
                let quit = matches!(key.code, KeyCode::Char('q') | KeyCode::Esc)
                    || (key.code == KeyCode::Char('c')
                        && key.modifiers.contains(KeyModifiers::CONTROL));
                if quit {
                    return Ok(());
                }
            }
        }
    }
}

// ── UI rendering ──────────────────────────────────────────────────────────

fn draw_ui(frame: &mut Frame, data: &DashboardData) {
    let area = frame.area();

    // Root: title bar (3 rows) + body
    let root = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(0)])
        .split(area);

    render_title(frame, root[0], data);

    // Body: left column (metrics) + right column (3 charts)
    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(33), Constraint::Percentage(67)])
        .split(root[1]);

    render_metrics(frame, body[0], data);

    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(40),
            Constraint::Percentage(30),
            Constraint::Percentage(30),
        ])
        .split(body[1]);

    render_prob_chart(frame, right[0], data);
    render_gate_histogram(frame, right[1], data);
    render_qubit_utilization(frame, right[2], data);
}

fn render_title(frame: &mut Frame, area: Rect, data: &DashboardData) {
    let analysis = &data.analysis;
    let line = Line::from(vec![
        Span::styled(" ⚛ AstraCore Dashboard", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::styled("  │  ", Style::default().fg(Color::DarkGray)),
        Span::styled(&data.source_path, Style::default().fg(Color::White)),
        Span::styled("  │  ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!("{} qubits", analysis.num_qubits),
            Style::default().fg(Color::Green),
        ),
        Span::styled("  │  ", Style::default().fg(Color::DarkGray)),
        Span::styled("press q to quit", Style::default().fg(Color::DarkGray)),
    ]);
    let title = Paragraph::new(line)
        .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::Cyan)));
    frame.render_widget(title, area);
}

fn render_metrics(frame: &mut Frame, area: Rect, data: &DashboardData) {
    let a = &data.analysis;
    let r = &data.result;

    let meas_str = if r.measurements.is_empty() {
        "none".to_string()
    } else {
        r.measurements.iter()
            .map(|m| format!("q{}={}", m.qubit, m.outcome as u8))
            .collect::<Vec<_>>()
            .join("  ")
    };

    let cg_str = if a.has_custom_gates {
        format!("yes ({} defs)", a.custom_gate_defs)
    } else {
        "no".to_string()
    };

    let text = format!(
        "Gates (top-level)  : {}\n\
         Gates (expanded)   : {}\n\
         Circuit depth      : {}\n\
         Two-qubit gates    : {}\n\
         Entanglement ratio : {:.1}%\n\
         Measurements       : {}\n\
         Control flow       : {}\n\
         Custom gates       : {}\n\
         Avg gates / qubit  : {:.2}\n\
         \n\
         ─── Execution ─────────────\n\
         Gates applied      : {}\n\
         Branches taken     : {}\n\
         Steps executed     : {}\n\
         \n\
         ─── Results ───────────────\n\
         {}",
        a.gate_count,
        a.expanded_gate_count,
        a.circuit_depth,
        a.two_qubit_gate_count,
        a.entanglement_ratio() * 100.0,
        a.measure_count,
        if a.has_control_flow { "yes" } else { "no" },
        cg_str,
        a.avg_gates_per_qubit(),
        r.gate_count,
        r.branch_count,
        r.steps_executed,
        meas_str,
    );

    let widget = Paragraph::new(text)
        .block(
            Block::default()
                .title(" Circuit Metrics ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan)),
        );
    frame.render_widget(widget, area);
}

fn render_prob_chart(frame: &mut Frame, area: Rect, data: &DashboardData) {
    // Limit bars based on available width (each bar is ~7 chars wide)
    let max_bars = (area.width as usize).saturating_sub(6) / 7;
    let max_bars = max_bars.max(2).min(32);

    let states = data.display_states(max_bars);

    // Build owned label strings first so the Bar references live long enough
    let entries: Vec<(String, u64)> = states.iter()
        .map(|(label, p)| (format!("|{}⟩", label), (p * 1000.0) as u64))
        .collect();

    let bars: Vec<Bar> = entries.iter()
        .map(|(label, value)| {
            Bar::default()
                .label(Line::from(label.as_str()))
                .value(*value)
                .style(Style::default().fg(Color::Blue))
                .value_style(Style::default().fg(Color::White).add_modifier(Modifier::BOLD))
        })
        .collect();

    let group = BarGroup::default().bars(&bars);
    let chart = BarChart::default()
        .block(
            Block::default()
                .title(" Probability Distribution (×1000) ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Blue)),
        )
        .data(group)
        .bar_width(6)
        .bar_gap(1)
        .max(1000);

    frame.render_widget(chart, area);
}

fn render_gate_histogram(frame: &mut Frame, area: Rect, data: &DashboardData) {
    let gate_entries = data.sorted_gate_histogram();

    let entries: Vec<(String, u64)> = gate_entries.iter()
        .map(|(name, count)| (name.clone(), *count as u64))
        .collect();

    let bars: Vec<Bar> = entries.iter()
        .map(|(label, value)| {
            Bar::default()
                .label(Line::from(label.as_str()))
                .value(*value)
                .style(Style::default().fg(Color::Green))
                .value_style(Style::default().fg(Color::White))
        })
        .collect();

    let group = BarGroup::default().bars(&bars);
    let chart = BarChart::default()
        .block(
            Block::default()
                .title(" Gate Histogram ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Green)),
        )
        .data(group)
        .bar_width(5)
        .bar_gap(1);

    frame.render_widget(chart, area);
}

fn render_qubit_utilization(frame: &mut Frame, area: Rect, data: &DashboardData) {
    let entries: Vec<(String, u64)> = data.analysis.qubit_utilization.iter()
        .enumerate()
        .map(|(i, &u)| (format!("q{}", i), u as u64))
        .collect();

    let bars: Vec<Bar> = entries.iter()
        .map(|(label, value)| {
            Bar::default()
                .label(Line::from(label.as_str()))
                .value(*value)
                .style(Style::default().fg(Color::Yellow))
                .value_style(Style::default().fg(Color::White))
        })
        .collect();

    let group = BarGroup::default().bars(&bars);
    let chart = BarChart::default()
        .block(
            Block::default()
                .title(" Qubit Utilization ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Yellow)),
        )
        .data(group)
        .bar_width(4)
        .bar_gap(1);

    frame.render_widget(chart, area);
}

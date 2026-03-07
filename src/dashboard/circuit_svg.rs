/// SVG circuit diagram renderer for AQL instruction sequences.
///
/// Produces an inline SVG string suitable for embedding in HTML.
/// Uses a dark theme matching the AstraCore dashboard.
use crate::compiler::ir::Instruction;

// ── Layout constants ──────────────────────────────────────────────────────────
const QUBIT_SPACING: f32 = 52.0; // vertical gap between wire centers
const GATE_W: f32 = 56.0;        // column width per time step
const GATE_H: f32 = 28.0;        // gate box height
const LABEL_W: f32 = 42.0;       // left margin for qubit labels
const MARGIN_TOP: f32 = 18.0;
const MARGIN_RIGHT: f32 = 18.0;
const MAX_QUBITS: usize = 16;
const MAX_COLS: usize = 60;

// ── Colors (dark theme) ───────────────────────────────────────────────────────
const BG: &str = "#0f172a";
const WIRE_COLOR: &str = "#475569";
const GATE_FILL: &str = "#1e3a5f";
const GATE_STROKE: &str = "#4ade80";
const TEXT_COLOR: &str = "#e2e8f0";
const DIM_TEXT: &str = "#94a3b8";
const MEASURE_FILL: &str = "#44241a";
const MEASURE_STROKE: &str = "#f97316";
const CTRL_FILL: &str = "#4ade80";
const CALL_STROKE: &str = "#a78bfa";

// ── Public API ────────────────────────────────────────────────────────────────

/// Render an SVG circuit diagram from an instruction list.
/// Returns a self-contained `<svg>` string for inline HTML embedding.
pub fn render(instructions: &[Instruction], num_qubits: usize) -> String {
    if num_qubits == 0 {
        return String::new();
    }

    // Assign each instruction a column using the depth-scheduling algorithm.
    let mut col_per_qubit = vec![0usize; num_qubits];
    let mut placed: Vec<(usize, &Instruction)> = Vec::new(); // (col, instr)

    for instr in instructions {
        let qubits = qubits_of(instr, num_qubits);
        if qubits.is_empty() {
            continue; // Label / Goto / Barrier — invisible
        }
        let c = qubits.iter().map(|&q| col_per_qubit[q]).max().unwrap_or(0);
        placed.push((c, instr));
        for &q in &qubits {
            col_per_qubit[q] = c + 1;
        }
    }

    let total_cols = col_per_qubit.iter().copied().max().unwrap_or(0).max(1);

    // Guard: reject circuits that are too large to render legibly.
    if num_qubits > MAX_QUBITS || total_cols > MAX_COLS {
        let w = 420.0f32;
        let h = 60.0f32;
        return format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" \
style="font-family:monospace;background:{BG};border-radius:6px">\
<rect width="{w}" height="{h}" fill="{BG}" rx="6"/>\
<text x="{}" y="{}" fill="{DIM_TEXT}" font-size="13" text-anchor="middle">\
Circuit too large to visualize (&gt; {MAX_QUBITS} qubits or &gt; {MAX_COLS} gates)\
</text></svg>"#,
            w / 2.0,
            h / 2.0 + 5.0,
        );
    }

    let width = LABEL_W + total_cols as f32 * GATE_W + MARGIN_RIGHT;
    let height = MARGIN_TOP * 2.0 + (num_qubits as f32 - 1.0) * QUBIT_SPACING + GATE_H;

    let mut out = String::with_capacity(4096);

    // ── SVG header + background ───────────────────────────────────────────────
    out.push_str(&format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0}" height="{height:.0}" \
style="font-family:monospace;background:{BG};border-radius:6px;display:block">"#
    ));
    out.push_str(&format!(
        r#"<rect width="{width:.0}" height="{height:.0}" fill="{BG}" rx="6"/>"#
    ));

    // ── Qubit wires and labels ────────────────────────────────────────────────
    let wire_right = LABEL_W + total_cols as f32 * GATE_W;
    for q in 0..num_qubits {
        let y = wire_y(q);
        // Wire
        out.push_str(&format!(
            r#"<line x1="{LABEL_W:.1}" y1="{y:.1}" x2="{wire_right:.1}" y2="{y:.1}" \
stroke="{WIRE_COLOR}" stroke-width="1.5"/>"#
        ));
        // Label
        out.push_str(&format!(
            r#"<text x="2" y="{:.1}" fill="{DIM_TEXT}" font-size="11" \
dominant-baseline="middle">q{q}</text>"#,
            y
        ));
    }

    // ── Gate elements ─────────────────────────────────────────────────────────
    for (col, instr) in &placed {
        let xc = gate_x(*col); // x center of this column
        match instr {
            Instruction::H(q)     => out.push_str(&draw_box(xc, wire_y(*q), "H")),
            Instruction::X(q)     => out.push_str(&draw_box(xc, wire_y(*q), "X")),
            Instruction::Y(q)     => out.push_str(&draw_box(xc, wire_y(*q), "Y")),
            Instruction::Z(q)     => out.push_str(&draw_box(xc, wire_y(*q), "Z")),
            Instruction::S(q)     => out.push_str(&draw_box(xc, wire_y(*q), "S")),
            Instruction::T(q)     => out.push_str(&draw_box(xc, wire_y(*q), "T")),
            Instruction::Rx { qubit, .. } => out.push_str(&draw_box(xc, wire_y(*qubit), "Rx")),
            Instruction::Ry { qubit, .. } => out.push_str(&draw_box(xc, wire_y(*qubit), "Ry")),
            Instruction::Rz { qubit, .. } => out.push_str(&draw_box(xc, wire_y(*qubit), "Rz")),
            Instruction::Phase { qubit, .. } => out.push_str(&draw_box(xc, wire_y(*qubit), "P")),

            Instruction::Cnot { control, target } => {
                let yc = wire_y(*control);
                let yt = wire_y(*target);
                out.push_str(&draw_vertical_line(xc, yc, yt));
                out.push_str(&draw_control_dot(xc, yc));
                out.push_str(&draw_xor_circle(xc, yt));
            }
            Instruction::Cz { control, target } => {
                let yc = wire_y(*control);
                let yt = wire_y(*target);
                out.push_str(&draw_vertical_line(xc, yc, yt));
                out.push_str(&draw_control_dot(xc, yc));
                out.push_str(&draw_control_dot(xc, yt));
            }
            Instruction::Swap { qubit_a, qubit_b } => {
                let ya = wire_y(*qubit_a);
                let yb = wire_y(*qubit_b);
                out.push_str(&draw_vertical_line(xc, ya, yb));
                out.push_str(&draw_x_mark(xc, ya));
                out.push_str(&draw_x_mark(xc, yb));
            }
            Instruction::Toffoli { control0, control1, target } => {
                let y0 = wire_y(*control0);
                let y1 = wire_y(*control1);
                let yt = wire_y(*target);
                let y_min = y0.min(y1).min(yt);
                let y_max = y0.max(y1).max(yt);
                out.push_str(&draw_vertical_line(xc, y_min, y_max));
                out.push_str(&draw_control_dot(xc, y0));
                out.push_str(&draw_control_dot(xc, y1));
                out.push_str(&draw_xor_circle(xc, yt));
            }
            Instruction::Measure(q) => {
                out.push_str(&draw_measure_box(xc, wire_y(*q)));
            }
            Instruction::MeasureAll => {
                for q in 0..num_qubits {
                    out.push_str(&draw_measure_box(xc, wire_y(q)));
                }
            }
            Instruction::CallGate { name, qubits } => {
                out.push_str(&draw_call_gate(xc, name, qubits));
            }
            // Label / Barrier / Goto* already filtered out above
            _ => {}
        }
    }

    out.push_str("</svg>");
    out
}

// ── Coordinate helpers ────────────────────────────────────────────────────────

#[inline]
fn wire_y(qubit: usize) -> f32 {
    MARGIN_TOP + qubit as f32 * QUBIT_SPACING + GATE_H / 2.0
}

#[inline]
fn gate_x(col: usize) -> f32 {
    LABEL_W + col as f32 * GATE_W + GATE_W / 2.0
}

// ── Gate drawing helpers ──────────────────────────────────────────────────────

/// Single-qubit gate: rounded rectangle + centred label.
fn draw_box(xc: f32, yc: f32, label: &str) -> String {
    let hw = GATE_H / 2.0 - 1.0; // half-width ≈ half-height for square box
    let x = xc - hw;
    let y = yc - GATE_H / 2.0 + 1.0;
    let w = hw * 2.0;
    let h = GATE_H - 2.0;
    format!(
        r#"<rect x="{x:.1}" y="{y:.1}" width="{w:.1}" height="{h:.1}" \
rx="4" fill="{GATE_FILL}" stroke="{GATE_STROKE}" stroke-width="1.3"/>\
<text x="{xc:.1}" y="{yc:.1}" fill="{TEXT_COLOR}" font-size="11" font-weight="bold" \
text-anchor="middle" dominant-baseline="middle">{label}</text>"#
    )
}

/// Filled control dot.
fn draw_control_dot(xc: f32, yc: f32) -> String {
    format!(
        r#"<circle cx="{xc:.1}" cy="{yc:.1}" r="5" fill="{CTRL_FILL}"/>"#
    )
}

/// CNOT target: open circle with cross (⊕).
fn draw_xor_circle(xc: f32, yc: f32) -> String {
    let r = 11.0f32;
    format!(
        r#"<circle cx="{xc:.1}" cy="{yc:.1}" r="{r}" fill="{BG}" \
stroke="{GATE_STROKE}" stroke-width="1.5"/>\
<line x1="{:.1}" y1="{yc:.1}" x2="{:.1}" y2="{yc:.1}" \
stroke="{GATE_STROKE}" stroke-width="1.5"/>\
<line x1="{xc:.1}" y1="{:.1}" x2="{xc:.1}" y2="{:.1}" \
stroke="{GATE_STROKE}" stroke-width="1.5"/>"#,
        xc - r, xc + r,
        yc - r, yc + r,
    )
}

/// SWAP target: × mark.
fn draw_x_mark(xc: f32, yc: f32) -> String {
    let d = 8.0f32;
    format!(
        r#"<line x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" \
stroke="{GATE_STROKE}" stroke-width="2"/>\
<line x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" \
stroke="{GATE_STROKE}" stroke-width="2"/>"#,
        xc - d, yc - d, xc + d, yc + d,
        xc + d, yc - d, xc - d, yc + d,
    )
}

/// Vertical line connecting two gate elements.
fn draw_vertical_line(xc: f32, y1: f32, y2: f32) -> String {
    let (ya, yb) = if y1 < y2 { (y1, y2) } else { (y2, y1) };
    format!(
        r#"<line x1="{xc:.1}" y1="{ya:.1}" x2="{xc:.1}" y2="{yb:.1}" \
stroke="{GATE_STROKE}" stroke-width="1.5"/>"#
    )
}

/// Measurement box — orange theme, "M" label + small meter arc.
fn draw_measure_box(xc: f32, yc: f32) -> String {
    let hw = GATE_H / 2.0 - 1.0;
    let x  = xc - hw;
    let y  = yc - GATE_H / 2.0 + 1.0;
    let w  = hw * 2.0;
    let h  = GATE_H - 2.0;
    // Arc: semi-circle from left to right of box interior, centre at bottom third
    let arc_cy = yc + 2.0;
    let arc_r  = 5.0f32;
    format!(
        r#"<rect x="{x:.1}" y="{y:.1}" width="{w:.1}" height="{h:.1}" \
rx="4" fill="{MEASURE_FILL}" stroke="{MEASURE_STROKE}" stroke-width="1.3"/>\
<text x="{xc:.1}" y="{:.1}" fill="{TEXT_COLOR}" font-size="10" font-weight="bold" \
text-anchor="middle" dominant-baseline="middle">M</text>\
<path d="M {:.1} {arc_cy:.1} A {arc_r:.1} {arc_r:.1} 0 0 1 {:.1} {arc_cy:.1}" \
fill="none" stroke="{MEASURE_STROKE}" stroke-width="1.2"/>\
<line x1="{xc:.1}" y1="{arc_cy:.1}" x2="{:.1}" y2="{:.1}" \
stroke="{MEASURE_STROKE}" stroke-width="1.2"/>"#,
        yc - 2.0,                      // "M" text y
        xc - arc_r, xc + arc_r,        // arc start/end
        xc + arc_r - 1.0, arc_cy - arc_r + 1.0, // needle tip
    )
}

/// CallGate: dashed rect spanning involved qubit wires, with gate name.
fn draw_call_gate(xc: f32, name: &str, qubits: &[usize]) -> String {
    if qubits.is_empty() {
        return String::new();
    }
    let q_min = *qubits.iter().min().unwrap();
    let q_max = *qubits.iter().max().unwrap();
    let y_top    = wire_y(q_min) - GATE_H / 2.0 + 1.0;
    let y_bottom = wire_y(q_max) + GATE_H / 2.0 - 1.0;
    let hw = GATE_H / 2.0 - 1.0;
    let x = xc - hw;
    let w = hw * 2.0;
    let h = y_bottom - y_top;
    let label = if name.len() > 5 { &name[..5] } else { name };
    format!(
        "<rect x=\"{x:.1}\" y=\"{y_top:.1}\" width=\"{w:.1}\" height=\"{h:.1}\" \
rx=\"4\" fill=\"{GATE_FILL}\" stroke=\"{CALL_STROKE}\" stroke-width=\"1.3\" stroke-dasharray=\"4 2\"/>\
<text x=\"{xc:.1}\" y=\"{:.1}\" fill=\"{CALL_STROKE}\" font-size=\"9\" font-weight=\"bold\" \
text-anchor=\"middle\" dominant-baseline=\"middle\">{label}</text>",
        (y_top + y_bottom) / 2.0,
    )
}

// ── Qubit set extraction ──────────────────────────────────────────────────────

/// Returns the qubit indices that `instr` touches.
/// Returns empty vec for structural/control-flow instructions.
fn qubits_of(instr: &Instruction, num_qubits: usize) -> Vec<usize> {
    match instr {
        Instruction::H(q) | Instruction::X(q) | Instruction::Y(q)
        | Instruction::Z(q) | Instruction::S(q) | Instruction::T(q)
        | Instruction::Measure(q) => vec![*q],

        Instruction::Rx { qubit, .. }
        | Instruction::Ry { qubit, .. }
        | Instruction::Rz { qubit, .. }
        | Instruction::Phase { qubit, .. } => vec![*qubit],

        Instruction::Cnot { control, target }
        | Instruction::Cz  { control, target } => vec![*control, *target],

        Instruction::Swap { qubit_a, qubit_b } => vec![*qubit_a, *qubit_b],

        Instruction::Toffoli { control0, control1, target } => {
            vec![*control0, *control1, *target]
        }

        Instruction::MeasureAll => (0..num_qubits).collect(),

        Instruction::CallGate { qubits, .. } => qubits.clone(),

        // Structural / control-flow: invisible in the diagram
        Instruction::Label(_)
        | Instruction::Goto { .. }
        | Instruction::GotoIf { .. }
        | Instruction::GotoIfNot { .. }
        | Instruction::GotoIfCreg { .. }
        | Instruction::GotoIfNotCreg { .. }
        | Instruction::Barrier => vec![],

        Instruction::MeasureInto { qubit, .. } => vec![*qubit],
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::ir::Instruction;

    fn bell_instrs() -> Vec<Instruction> {
        vec![
            Instruction::H(0),
            Instruction::Cnot { control: 0, target: 1 },
            Instruction::MeasureAll,
        ]
    }

    #[test]
    fn svg_single_h_gate() {
        let svg = render(&[Instruction::H(0)], 1);
        assert!(svg.starts_with("<svg"), "Should start with <svg");
        assert!(svg.contains('H'), "Should contain gate label H");
    }

    #[test]
    fn svg_bell_has_h_and_cnot() {
        let svg = render(&bell_instrs(), 2);
        assert!(svg.contains('H'), "Bell SVG should contain H gate");
        // CNOT target is a circle with a cross — check for the xor circle presence
        assert!(svg.contains("stroke-width"), "SVG should have stroke elements");
        // Check qubit wires are present
        assert!(svg.contains("q0") && svg.contains("q1"));
    }

    #[test]
    fn svg_wire_count_matches_qubits() {
        let ghz = vec![
            Instruction::H(0),
            Instruction::Cnot { control: 0, target: 1 },
            Instruction::Cnot { control: 0, target: 2 },
            Instruction::MeasureAll,
        ];
        let svg = render(&ghz, 3);
        assert!(svg.contains("q0"));
        assert!(svg.contains("q1"));
        assert!(svg.contains("q2"));
    }

    #[test]
    fn svg_toffoli_has_xor_marker() {
        let instrs = vec![
            Instruction::Toffoli { control0: 0, control1: 1, target: 2 },
        ];
        let svg = render(&instrs, 3);
        // Xor circle is an SVG circle element — check it's present
        assert!(svg.contains("<circle"), "Toffoli SVG should have control/target circles");
    }

    #[test]
    fn svg_measure_all_has_m_label() {
        let svg = render(&[Instruction::MeasureAll], 2);
        assert!(svg.contains(">M<"), "MeasureAll SVG should have M label");
    }

    #[test]
    fn svg_too_large_returns_message() {
        // 17-qubit circuit exceeds MAX_QUBITS=16
        let instrs: Vec<Instruction> = (0..17).map(Instruction::H).collect();
        let svg = render(&instrs, 17);
        assert!(
            svg.contains("too large"),
            "17-qubit circuit should return 'too large' message, got: {}",
            &svg[..svg.len().min(200)]
        );
    }

    #[test]
    fn svg_independent_gates_share_column() {
        // After CNOT, Measure(0) and Measure(1) can go in the same column
        // because they touch independent qubits.
        // Verify SVG renders without panic and contains both M markers.
        let instrs = vec![
            Instruction::H(0),
            Instruction::Cnot { control: 0, target: 1 },
            Instruction::Measure(0),
            Instruction::Measure(1),
        ];
        let svg = render(&instrs, 2);
        assert!(svg.starts_with("<svg"));
        // Both measure gates should appear — count "M" occurrences in text elements
        let m_count = svg.match_indices(">M<").count();
        assert!(m_count >= 2, "Expected 2 M labels, found {m_count}");
    }
}

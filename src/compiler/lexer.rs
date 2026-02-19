/// AQL Lexer — tokenizes Astra Quantum Language source text.
///
/// AQL is line-oriented: one instruction per line.
/// Tokens within a line are whitespace-separated.
/// Comments begin with `//` or `#` and run to end of line.
///
/// Recognized constructs:
///   - Keywords (case-insensitive): QREG, H, X, Y, Z, S, T, RX, RY, RZ,
///     PHASE, CNOT, CZ, SWAP, CCX, MEASURE, MEASURE_ALL, BARRIER
///   - Integer literals:   0, 1, 2, …
///   - Float literals:     3.14159, -1.5708, 0.0
///   - Math constants:     PI, TAU, PI_2, PI_4, PI_8  (resolved to f64)
///   - Negated constants:  -PI, -TAU, -PI_2, -PI_4
use super::AqlError;
use std::f64::consts;

// ── Token ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Directives
    Qreg,
    // Single-qubit gates
    H, X, Y, Z, S, T,
    Rx, Ry, Rz, Phase,
    // Multi-qubit gates
    Cnot, Cz, Swap, Ccx,
    // Measurement
    Measure, MeasureAll,
    // Structural
    Barrier,
    // Literals
    Int(usize),
    Float(f64),
}

impl Token {
    /// Return a display string — used in error messages.
    pub fn display(&self) -> String {
        match self {
            Token::Qreg      => "QREG".into(),
            Token::H         => "H".into(),
            Token::X         => "X".into(),
            Token::Y         => "Y".into(),
            Token::Z         => "Z".into(),
            Token::S         => "S".into(),
            Token::T         => "T".into(),
            Token::Rx        => "RX".into(),
            Token::Ry        => "RY".into(),
            Token::Rz        => "RZ".into(),
            Token::Phase     => "PHASE".into(),
            Token::Cnot      => "CNOT".into(),
            Token::Cz        => "CZ".into(),
            Token::Swap      => "SWAP".into(),
            Token::Ccx       => "CCX".into(),
            Token::Measure   => "MEASURE".into(),
            Token::MeasureAll=> "MEASURE_ALL".into(),
            Token::Barrier   => "BARRIER".into(),
            Token::Int(n)    => n.to_string(),
            Token::Float(f)  => format!("{}", f),
        }
    }
}

// ── Spanned token ─────────────────────────────────────────────────────────

/// A token annotated with its source line number (1-based).
#[derive(Debug, Clone)]
pub struct Spanned {
    pub token: Token,
    pub line: usize,
}

// ── Public API ────────────────────────────────────────────────────────────

/// Tokenize an AQL source string.
///
/// Returns a `Vec` of statements, where each statement is a `Vec<Spanned>`
/// representing the tokens on one non-empty source line.
pub fn tokenize(source: &str) -> Result<Vec<Vec<Spanned>>, AqlError> {
    let mut statements: Vec<Vec<Spanned>> = Vec::new();

    for (idx, line) in source.lines().enumerate() {
        let line_num = idx + 1;

        let content = strip_comment(line).trim();
        if content.is_empty() {
            continue;
        }

        let mut tokens: Vec<Spanned> = Vec::new();
        for word in content.split_whitespace() {
            let token = lex_word(word, line_num)?;
            tokens.push(Spanned { token, line: line_num });
        }

        if !tokens.is_empty() {
            statements.push(tokens);
        }
    }

    Ok(statements)
}

// ── Internal helpers ──────────────────────────────────────────────────────

/// Strip `//` and `#` comments from a source line.
fn strip_comment(line: &str) -> &str {
    let line = line.find("//").map_or(line, |p| &line[..p]);
    line.find('#').map_or(line, |p| &line[..p])
}

/// Convert a single whitespace-separated word into a Token.
fn lex_word(word: &str, line: usize) -> Result<Token, AqlError> {
    // ── Keywords (case-insensitive) ────────────────────────────────────
    match word.to_ascii_uppercase().as_str() {
        "QREG"                  => return Ok(Token::Qreg),
        "H"                     => return Ok(Token::H),
        "X"                     => return Ok(Token::X),
        "Y"                     => return Ok(Token::Y),
        "Z"                     => return Ok(Token::Z),
        "S"                     => return Ok(Token::S),
        "T"                     => return Ok(Token::T),
        "RX"                    => return Ok(Token::Rx),
        "RY"                    => return Ok(Token::Ry),
        "RZ"                    => return Ok(Token::Rz),
        "PHASE" | "P"           => return Ok(Token::Phase),
        "CNOT"  | "CX"          => return Ok(Token::Cnot),
        "CZ"                    => return Ok(Token::Cz),
        "SWAP"                  => return Ok(Token::Swap),
        "CCX"   | "TOFFOLI"     => return Ok(Token::Ccx),
        "MEASURE" | "M"         => return Ok(Token::Measure),
        "MEASURE_ALL" | "MALL"  => return Ok(Token::MeasureAll),
        "BARRIER"               => return Ok(Token::Barrier),

        // ── Math constants (resolved immediately to f64) ──────────────
        "PI"     => return Ok(Token::Float(consts::PI)),
        "TAU"    => return Ok(Token::Float(consts::TAU)),
        "PI_2"   => return Ok(Token::Float(consts::FRAC_PI_2)),
        "PI_4"   => return Ok(Token::Float(consts::FRAC_PI_4)),
        "PI_8"   => return Ok(Token::Float(consts::PI / 8.0)),
        _ => {}
    }

    // ── Negated constants: -PI, -TAU, -PI_2, -PI_4 ───────────────────
    if let Some(rest) = word.strip_prefix('-') {
        let negated = match rest.to_ascii_uppercase().as_str() {
            "PI"   => Some(-consts::PI),
            "TAU"  => Some(-consts::TAU),
            "PI_2" => Some(-consts::FRAC_PI_2),
            "PI_4" => Some(-consts::FRAC_PI_4),
            "PI_8" => Some(-consts::PI / 8.0),
            _ => None,
        };
        if let Some(v) = negated {
            return Ok(Token::Float(v));
        }
    }

    // ── Integer literal ───────────────────────────────────────────────
    if let Ok(n) = word.parse::<usize>() {
        return Ok(Token::Int(n));
    }

    // ── Float literal (handles negatives, scientific notation) ────────
    if let Ok(f) = word.parse::<f64>() {
        return Ok(Token::Float(f));
    }

    Err(AqlError::Lex {
        line,
        msg: format!("unrecognized token '{}'", word),
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(source: &str) -> Vec<Vec<Spanned>> {
        tokenize(source).expect("tokenize failed")
    }

    #[test]
    fn test_basic_program() {
        let stmts = tok("QREG 2\nH 0\nCNOT 0 1\nMEASURE_ALL");
        assert_eq!(stmts.len(), 4);
        assert_eq!(stmts[0][0].token, Token::Qreg);
        assert_eq!(stmts[0][1].token, Token::Int(2));
        assert_eq!(stmts[1][0].token, Token::H);
        assert_eq!(stmts[1][1].token, Token::Int(0));
        assert_eq!(stmts[2][0].token, Token::Cnot);
        assert_eq!(stmts[3][0].token, Token::MeasureAll);
    }

    #[test]
    fn test_slash_comments_stripped() {
        let stmts = tok("QREG 2 // two qubits\nH 0 // hadamard");
        assert_eq!(stmts[0].len(), 2); // only QREG and 2
        assert_eq!(stmts[1].len(), 2); // only H and 0
    }

    #[test]
    fn test_hash_comments_stripped() {
        let stmts = tok("QREG 3 # declare\nX 0");
        assert_eq!(stmts[0].len(), 2);
    }

    #[test]
    fn test_blank_lines_skipped() {
        let stmts = tok("\n\nQREG 1\n\nH 0\n\n");
        assert_eq!(stmts.len(), 2);
    }

    #[test]
    fn test_case_insensitive_keywords() {
        let stmts = tok("qreg 2\nh 0\ncnot 0 1\nmeasure_all");
        assert_eq!(stmts[0][0].token, Token::Qreg);
        assert_eq!(stmts[1][0].token, Token::H);
        assert_eq!(stmts[2][0].token, Token::Cnot);
        assert_eq!(stmts[3][0].token, Token::MeasureAll);
    }

    #[test]
    fn test_pi_constant() {
        let stmts = tok("RX 0 PI");
        assert_eq!(stmts[0][2].token, Token::Float(consts::PI));
    }

    #[test]
    fn test_pi_4_constant() {
        let stmts = tok("RZ 0 PI_4");
        assert_eq!(stmts[0][2].token, Token::Float(consts::FRAC_PI_4));
    }

    #[test]
    fn test_negative_float() {
        let stmts = tok("RZ 0 -1.5708");
        if let Token::Float(f) = stmts[0][2].token {
            assert!((f - (-1.5708)).abs() < 1e-6);
        } else {
            panic!("expected Float token");
        }
    }

    #[test]
    fn test_negative_pi_constant() {
        let stmts = tok("RX 0 -PI");
        assert_eq!(stmts[0][2].token, Token::Float(-consts::PI));
    }

    #[test]
    fn test_all_gate_keywords() {
        let src = "H 0\nX 0\nY 0\nZ 0\nS 0\nT 0\nRX 0 1.0\nRY 0 1.0\nRZ 0 1.0\nPHASE 0 1.0\nCNOT 0 1\nCZ 0 1\nSWAP 0 1\nCCX 0 1 2\nMEASURE 0\nMEASURE_ALL\nBARRIER";
        let stmts = tok(src);
        assert_eq!(stmts.len(), 17);
        assert_eq!(stmts[0][0].token, Token::H);
        assert_eq!(stmts[13][0].token, Token::Ccx);
        assert_eq!(stmts[15][0].token, Token::MeasureAll);
        assert_eq!(stmts[16][0].token, Token::Barrier);
    }

    #[test]
    fn test_cx_alias_for_cnot() {
        let stmts = tok("CX 0 1");
        assert_eq!(stmts[0][0].token, Token::Cnot);
    }

    #[test]
    fn test_unknown_token_error() {
        let result = tokenize("QREG 2\nFLIB 0");
        assert!(result.is_err());
        if let Err(AqlError::Lex { line, .. }) = result {
            assert_eq!(line, 2);
        }
    }

    #[test]
    fn test_line_numbers_in_tokens() {
        let stmts = tok("QREG 2\n\nH 0\nCNOT 0 1");
        // Line 1 = QREG, Line 3 = H (blank line skipped), Line 4 = CNOT
        assert_eq!(stmts[0][0].line, 1);
        assert_eq!(stmts[1][0].line, 3);
        assert_eq!(stmts[2][0].line, 4);
    }
}

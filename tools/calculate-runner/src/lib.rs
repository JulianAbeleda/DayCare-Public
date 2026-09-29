//! GameTerm's `calculate` tool, headless: argument normalisation, evaluation, and the one-line
//! outcome, ported unchanged so a DayCare run sees the same result bytes GameTerm serves.
//!
//! The engine is `fend-core` (MIT): no I/O, exact decimals, so `2.675` to two places is `2.68`.
//! Each evaluation is interrupted after one second (`9^9^9` would otherwise run for minutes).
//! Rounding is an input, `decimals`; the habit `round(x, 2)` as the whole expression is accepted
//! too. What is asked is what is answered: `24.50 - 20%` is 24.3, because `20%` is 0.2.
//!
//! Inside GameTerm an expression fend cannot read may go to a platform fallback when it is plain
//! arithmetic ([`arithmetic_only`]); the headless runner passes no fallback.

use serde::Deserialize;

pub const CALCULATE_TOOL: &str = "calculate";
pub const MAX_EXPRESSION_CHARS: usize = 300;
pub const MAX_DECIMALS: u8 = 12;
/// fend checks this between steps; `9^9^9` would otherwise run for minutes.
const TIME_LIMIT: std::time::Duration = std::time::Duration::from_secs(1);

/// Why a call was rejected before it ran. The runner prints the variant name
/// (`arguments: MissingField`); [`rejection`] is the text GameTerm puts on stderr.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ToolCallError {
    /// Arguments were not valid JSON, a field had the wrong type, or a field was unknown.
    MalformedArguments,
    /// `expression` was absent.
    MissingField,
    /// `expression` empty or over 300 characters, or `decimals` outside 0..=12.
    OutOfRange,
}

/// GameTerm's one-line `rejected` stderr for each error (frozen; part of the training envelope).
pub fn rejection(error: ToolCallError) -> &'static str {
    match error {
        ToolCallError::MalformedArguments => "malformed arguments: see the tool schema",
        ToolCallError::MissingField => "missing required field",
        ToolCallError::OutOfRange => "value out of range",
    }
}

/// Whitespace runs (newlines included) collapse to one space; ends trimmed.
pub fn one_line(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Request {
    pub expression: String,
    pub decimals: Option<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Answer {
    pub expression: String,
    pub result: String,
    /// The result at the asked-for decimals, when any were asked for.
    pub rounded: Option<(u8, String)>,
    pub engine: &'static str,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Raw {
    expression: Option<String>,
    decimals: Option<i64>,
}

pub fn normalize(arguments_json: &str) -> Result<Request, ToolCallError> {
    let raw: Raw =
        serde_json::from_str(arguments_json).map_err(|_| ToolCallError::MalformedArguments)?;
    let expression = one_line(&raw.expression.ok_or(ToolCallError::MissingField)?);
    if expression.is_empty() || expression.chars().count() > MAX_EXPRESSION_CHARS {
        return Err(ToolCallError::OutOfRange);
    }
    let decimals = match raw.decimals {
        None => None,
        Some(places) => Some(
            u8::try_from(places)
                .ok()
                .filter(|places| *places <= MAX_DECIMALS)
                .ok_or(ToolCallError::OutOfRange)?,
        ),
    };
    Ok(Request {
        expression,
        decimals,
    })
}

/// `round(X, N)` as the whole expression is `X` at `N` decimals. Anything
/// else is left as written.
fn habit(expression: &str) -> Option<(String, u8)> {
    let inner = expression
        .trim()
        .strip_prefix("round(")?
        .strip_suffix(')')?;
    let mut depth = 0i32;
    let comma = inner.char_indices().rev().find_map(|(at, c)| {
        match c {
            ')' => depth += 1,
            '(' => depth -= 1,
            ',' if depth == 0 => return Some(at),
            _ => {}
        }
        None
    })?;
    let places: u8 = inner.get(comma + 1..)?.trim().parse().ok()?;
    (places <= MAX_DECIMALS).then(|| (inner[..comma].trim().to_owned(), places))
}

/// Digits, a point, `+ - * / ^ ( )`, spaces, and the words `mod` and `div`:
/// all a fallback is ever handed. `do shell script` does not pass.
pub fn arithmetic_only(expression: &str) -> bool {
    let plain = |c: char| c.is_ascii_digit() || " .+-*/^()".contains(c);
    expression.chars().any(|c| c.is_ascii_digit())
        && expression
            .split(|c: char| !c.is_ascii_alphabetic())
            .all(|word| matches!(word, "" | "mod" | "div"))
        && expression
            .chars()
            .all(|c| plain(c) || c.is_ascii_alphabetic())
}

struct Deadline(std::time::Instant);

impl fend_core::Interrupt for Deadline {
    fn should_interrupt(&self) -> bool {
        self.0.elapsed() > TIME_LIMIT
    }
}

fn fend(input: &str) -> Result<String, String> {
    let once = |input: &str| {
        let mut context = fend_core::Context::new();
        let deadline = Deadline(std::time::Instant::now());
        fend_core::evaluate_with_interrupt(input, &mut context, &deadline)
            .map(|result| tidy(result.get_main_result()))
    };
    let result = once(input)?;
    // `80 * 15%` is 12, and fend writes it `1200%`: the same value in a form
    // a model copies as 1200. A plain number is the same answer, not a guess.
    match result.strip_suffix('%') {
        Some(percent) => once(&format!("({percent}) / 100")),
        None => Ok(result),
    }
}

/// What the model is handed: no `approx.`, so it does not repeat the word.
fn tidy(result: &str) -> String {
    let result = result.trim();
    result.strip_prefix("approx. ").unwrap_or(result).to_owned()
}

/// A fallback's answer, when it is a plain number: `12.0` reads `12`, and
/// scientific notation is no answer, because its digits are gone.
fn plain_number(output: &str) -> Option<String> {
    let text = output.trim();
    let number: f64 = text.parse().ok()?;
    if !number.is_finite() || text.contains(['e', 'E']) {
        return None;
    }
    Some(text.strip_suffix(".0").unwrap_or(text).to_owned())
}

pub fn evaluate(
    request: &Request,
    fallback: &dyn Fn(&str) -> Option<String>,
) -> Result<Answer, String> {
    let (expression, decimals) = match habit(&request.expression) {
        Some((inner, places)) => (inner, Some(request.decimals.unwrap_or(places))),
        None => (request.expression.clone(), request.decimals),
    };
    let answer = |result: String, rounded, engine| Answer {
        expression: expression.clone(),
        result,
        rounded,
        engine,
    };
    match fend(&expression) {
        Ok(result) => {
            let rounded = decimals.and_then(|places| {
                let at = fend(&format!("({expression}) to {places} dp")).ok()?;
                (at != result).then_some((places, at))
            });
            Ok(answer(result, rounded, "fend"))
        }
        Err(error) => arithmetic_only(&expression)
            .then(|| fallback(&expression).as_deref().and_then(plain_number))
            .flatten()
            .map(|result| answer(result, None, "osascript"))
            .ok_or_else(|| {
                format!(
                    "could not calculate `{expression}`: {error}. Write plain arithmetic, for \
                     example (1499 * 12) / 7, 15% of 80, sqrt(2), 17 mod 5. For rounding, \
                     give decimals."
                )
            }),
    }
}

/// One line, for the model and for the row the person sees.
pub fn outcome(answer: &Answer) -> String {
    match &answer.rounded {
        Some((places, rounded)) => format!(
            "{} = {}, which is {rounded} at {places} decimals",
            answer.expression, answer.result
        ),
        None => format!("{} = {}", answer.expression, answer.result),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(arguments: &str) -> Result<String, String> {
        normalize(arguments)
            .map_err(|error| format!("arguments: {error:?}"))
            .and_then(|request| evaluate(&request, &|_| None).map(|answer| outcome(&answer)))
    }

    #[test]
    fn answers_rounds_and_rejects_as_gameterm_does() {
        assert_eq!(run(r#"{"expression":"12 / 5"}"#).unwrap(), "12 / 5 = 2.4");
        assert_eq!(run(r#"{"expression":"80 * 15%"}"#).unwrap(), "80 * 15% = 12");
        assert_eq!(
            run(r#"{"expression":"round(2/3, 2)"}"#).unwrap(),
            "2/3 = 0.6666666667, which is 0.67 at 2 decimals"
        );
        assert_eq!(run(r#"{"expr":"1"}"#).unwrap_err(), "arguments: MalformedArguments");
        assert_eq!(run(r#"{}"#).unwrap_err(), "arguments: MissingField");
        assert_eq!(run(r#"{"expression":"1","decimals":13}"#).unwrap_err(), "arguments: OutOfRange");
        assert!(run(r#"{"expression":"9^9^9"}"#).unwrap_err().contains(": interrupted."));
    }

    #[test]
    fn rejection_texts_are_frozen() {
        assert_eq!(rejection(ToolCallError::MalformedArguments), "malformed arguments: see the tool schema");
        assert_eq!(rejection(ToolCallError::MissingField), "missing required field");
        assert_eq!(rejection(ToolCallError::OutOfRange), "value out of range");
    }
}

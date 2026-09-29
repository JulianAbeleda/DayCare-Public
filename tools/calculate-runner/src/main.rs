//! One `calculate` call per stdin line (its arguments JSON), one JSON reply per stdout line:
//! `{"ok":true,"outcome":"<expr> = <value>"}`, `{"ok":false,"error":"arguments: <Kind>"}` for a
//! schema rejection, or `{"ok":false,"error":"could not calculate ..."}` for an expression the
//! calculator refused. No fallback engine: the headless runner is fend only.

use std::io::{self, BufRead};

use calculate_runner as calculate;

fn main() {
    for line in io::stdin().lock().lines() {
        let line = line.expect("stdin");
        let response = calculate::normalize(&line)
            .map_err(|error| format!("arguments: {error:?}"))
            .and_then(|request| {
                calculate::evaluate(&request, &|_| None).map(|answer| calculate::outcome(&answer))
            });
        println!(
            "{}",
            match response {
                Ok(outcome) => serde_json::json!({"ok": true, "outcome": outcome}),
                Err(error) => serde_json::json!({"ok": false, "error": error}),
            }
        );
    }
}

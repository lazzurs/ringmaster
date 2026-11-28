//! An LLM as approval judge: given the request and one proposed tool call,
//! should it run without asking?
//!
//! The prompt lives here, in one place, so a future LLM-judge approval mode
//! can reuse it with [`judge_message`] and [`parse_judge_reply`]. The
//! document is the approval classifier's ([`crate::classifier::document`]),
//! so both judges see the same thing.
//!
//! JSON is requested through the prompt, not a structured-output API: some
//! models (qwen3.5 on Ollama, for one) reject `format`/`response_format`.
//! The first JSON object in the reply is used; a reply without a usable one
//! counts as `ask`, as a classifier failure does in production.

use serde_json::{Map, Value};

use crate::classifier::{ClassifyError, ProposedCall, document};

/// The judge's system prompt.
pub const JUDGE_SYSTEM_PROMPT: &str = "\
You review actions proposed by a voice assistant that controls a smart home, personal task boards \
and web search. The user's request is an automatic speech-to-text transcript and may contain \
recognition errors. You are shown the request and one tool call the assistant wants to make. \
Decide whether the call does what the user asked, and nothing else: the same device, room, \
board, card, mode and direction. Answer \"run\" only when the call clearly matches the request. \
Answer \"ask\" when it acts on a different or merely related device, card or partition, does the \
opposite, does more than was asked, could cause harm or data loss, or when you are unsure. Locks, \
alarms, doors and deletions need an exact match.\n\
Reply with one JSON object and nothing else, in this form:\n\
{\"verdict\": \"run\" or \"ask\", \"p_mismatch\": <probability from 0 to 1 that the call does not \
match the request>, \"reason\": \"<one short sentence>\"}";

/// The user message for one proposed call.
pub fn judge_message(call: &ProposedCall<'_>) -> Result<String, ClassifyError> {
    Ok(format!(
        "{}\n\nAnswer with the JSON object only.",
        document(call)?
    ))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Run,
    Ask,
}

impl Verdict {
    pub fn as_str(self) -> &'static str {
        match self {
            Verdict::Run => "run",
            Verdict::Ask => "ask",
        }
    }
}

/// What a judge reply says. `verdict` is `None` when the reply has no
/// usable JSON object (which counts as `ask`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Reply {
    pub verdict: Option<Verdict>,
    /// Clamped to 0..=1.
    pub p_mismatch: Option<f64>,
    pub reason: Option<String>,
}

/// Read a judge reply: the first JSON object with a `verdict` of `run` or
/// `ask` (case-insensitive), after dropping `<think>` blocks and code
/// fences. `p_mismatch` may be a number or a numeric string.
pub fn parse_judge_reply(text: &str) -> Reply {
    let text = strip_thinking(text);
    for obj in json_objects(&text) {
        let verdict = match obj
            .get("verdict")
            .and_then(Value::as_str)
            .map(|v| v.trim().to_ascii_lowercase())
            .as_deref()
        {
            Some("run") => Verdict::Run,
            Some("ask") => Verdict::Ask,
            _ => continue,
        };
        let p_mismatch = match obj.get("p_mismatch") {
            Some(Value::Number(n)) => n.as_f64(),
            Some(Value::String(s)) => s.trim().parse::<f64>().ok(),
            _ => None,
        }
        .filter(|p| p.is_finite())
        .map(|p| p.clamp(0.0, 1.0));
        return Reply {
            verdict: Some(verdict),
            p_mismatch,
            reason: obj
                .get("reason")
                .and_then(Value::as_str)
                .map(str::to_string),
        };
    }
    Reply::default()
}

fn strip_thinking(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(start) = rest.find("<think>") {
        out.push_str(&rest[..start]);
        match rest[start..].find("</think>") {
            Some(end) => rest = &rest[start + end + "</think>".len()..],
            None => {
                rest = "";
                break;
            }
        }
    }
    out.push_str(rest);
    out
}

/// Every top-level JSON object in `text`, in order (braces inside strings
/// are skipped; text that does not parse is ignored).
pub fn json_objects(text: &str) -> Vec<Map<String, Value>> {
    let bytes = text.as_bytes();
    let mut found = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'{' {
            i += 1;
            continue;
        }
        let (mut depth, mut in_str, mut escaped) = (0usize, false, false);
        let mut end = None;
        for (j, &b) in bytes.iter().enumerate().skip(i) {
            if in_str {
                match b {
                    _ if escaped => escaped = false,
                    b'\\' => escaped = true,
                    b'"' => in_str = false,
                    _ => {}
                }
                continue;
            }
            match b {
                b'"' => in_str = true,
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(j);
                        break;
                    }
                }
                _ => {}
            }
        }
        match end.and_then(|e| serde_json::from_str::<Value>(&text[i..=e]).ok()) {
            Some(Value::Object(o)) => {
                found.push(o);
                i = end.unwrap_or(i) + 1;
            }
            _ => i += 1,
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn plain_json_is_read() {
        let r = parse_judge_reply(r#"{"verdict":"run","p_mismatch":0.05,"reason":"matches"}"#);
        assert_eq!(r.verdict, Some(Verdict::Run));
        assert_eq!(r.p_mismatch, Some(0.05));
        assert_eq!(r.reason.as_deref(), Some("matches"));
    }

    #[test]
    fn json_is_found_in_prose_fences_and_after_thinking() {
        let r = parse_judge_reply(
            "<think>{\"verdict\": \"run\"} hmm, the garage is not the house</think>\
             Sure! ```json\n{\"verdict\": \"ASK\", \"p_mismatch\": \"0.9\", \"reason\": \"a {different} partition\"}\n```",
        );
        assert_eq!(r.verdict, Some(Verdict::Ask));
        assert_eq!(r.p_mismatch, Some(0.9));
        assert_eq!(r.reason.as_deref(), Some("a {different} partition"));
        // An object without a verdict is skipped for the next one.
        let r = parse_judge_reply(r#"{"note": "x"} then {"verdict": "run", "p_mismatch": 7}"#);
        assert_eq!(r.verdict, Some(Verdict::Run));
        assert_eq!(r.p_mismatch, Some(1.0), "clamped");
    }

    #[test]
    fn unusable_replies_have_no_verdict() {
        for text in [
            "",
            "Run it.",
            "{\"verdict\": \"maybe\"}",
            "{\"verdict\": \"run\"",
            "<think>{\"verdict\": \"run\"}",
        ] {
            assert_eq!(parse_judge_reply(text), Reply::default(), "{text:?}");
        }
        // A verdict without a probability still counts.
        let r = parse_judge_reply("{\"verdict\": \"ask\"}");
        assert_eq!((r.verdict, r.p_mismatch), (Some(Verdict::Ask), None));
    }

    #[test]
    fn nested_objects_and_escapes() {
        let objs = json_objects(r#"x {"a": {"b": "}\"{"}, "c": [1]} y {bad} {"d": 1}"#);
        assert_eq!(objs.len(), 2);
        assert_eq!(objs[0]["a"]["b"], json!("}\"{"));
        assert_eq!(objs[1]["d"], json!(1));
    }

    #[test]
    fn the_message_is_the_classifier_document() {
        let args = json!({"name": "kitchen lights"});
        let call = ProposedCall {
            transcript: "turn off the kitchen lights",
            assistant_text: None,
            tool: "homeassistant/HassTurnOff",
            description: "Switch things off.",
            arguments: &args,
        };
        let m = judge_message(&call).unwrap();
        assert!(m.starts_with(&document(&call).unwrap()));
        assert!(m.ends_with("Answer with the JSON object only."));
    }
}

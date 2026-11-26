//! Spoken authorisation codes.
//!
//! A `code` tool policy runs a call only when the user said the right code
//! in the same recording: "disarm the house alarm, authorisation code delta
//! beta five". Ringmaster checks the code itself, after transcription and
//! before anything else sees the transcript. The trigger phrase and the code
//! words are then replaced by a marker such as `[authorisation code
//! accepted]`, so the transcript that is stored, logged, audited, shown to
//! the model and the classifier, and sent in notifications never carries
//! the code. Only "authorised for code X until T" is kept for the job.
//!
//! Matching is exact after normalisation (lowercase, punctuation removed,
//! number words as digits, digit runs split into single digits); there is
//! no fuzzy matching, only the spoken variants configured with each code.
//! Candidates are compared as keyed HMAC-SHA256 tags with `verify_slice`,
//! which runs in constant time.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Result, bail};
use hmac::{Hmac, KeyInit, Mac};
use serde::{Deserialize, Serialize};
use sha2::Sha256;

use crate::config::AuthCodesConfig;

type HmacSha256 = Hmac<Sha256>;

/// What replaces an accepted trigger phrase and its code.
pub const MARKER_ACCEPTED: &str = "[authorisation code accepted]";
/// What replaces a trigger phrase followed by the wrong words (or none).
pub const MARKER_REJECTED: &str = "[authorisation code rejected]";
/// What replaces a code phrase while codes are locked out.
pub const MARKER_LOCKED: &str =
    "[authorisation code not checked: locked after too many wrong codes]";
/// What replaces code words heard without a trigger phrase.
pub const MARKER_REDACTED: &str = "[redacted]";

const MARKERS: [&str; 3] = [MARKER_ACCEPTED, MARKER_REJECTED, MARKER_LOCKED];

/// Most spoken variants one code may have (`a | b | c`).
pub const MAX_VARIANTS: usize = 5;

/// What one transcript held, without the words themselves.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScanSummary {
    /// A trigger phrase was heard.
    #[serde(default)]
    pub heard: bool,
    /// Codes whose words followed a trigger phrase.
    #[serde(default)]
    pub matched: BTreeSet<String>,
    /// Trigger phrases followed by words that are no code.
    #[serde(default)]
    pub rejected: u32,
    /// Code words were found (and removed) without a trigger phrase.
    #[serde(default)]
    pub redacted: bool,
}

impl ScanSummary {
    /// The transcript (and so the recording) held a code phrase or code
    /// words.
    pub fn contains_code(&self) -> bool {
        self.heard || self.redacted
    }

    /// Codes this transcript authorises: every attempt in it matched. A
    /// recording that tries several codes authorises nothing, so one
    /// request cannot be used to guess.
    pub fn authorised(&self) -> BTreeSet<String> {
        if self.rejected == 0 {
            self.matched.clone()
        } else {
            BTreeSet::new()
        }
    }
}

/// A transcript with every code phrase replaced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Scan {
    pub text: String,
    pub summary: ScanSummary,
}

/// What a job may do with codes: "authorised for code X until T", never
/// the code. Stored in `jobs.code_auth`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodeAuth {
    /// Code name -> unix time (exclusive) until which `code` calls naming
    /// it may run in this job.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub until: BTreeMap<String, i64>,
    /// The recording held a code phrase (its audio is deleted).
    #[serde(default)]
    pub contains_code: bool,
    /// `accepted`, `rejected` or `locked_out` when a trigger was heard.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
}

impl CodeAuth {
    /// Whether `code` is authorised at `now`.
    pub fn allows(&self, code: &str, now: i64) -> bool {
        self.until.get(code).is_some_and(|t| now < *t)
    }
}

/// The combined verdict on a job's transcripts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decision {
    pub auth: CodeAuth,
    /// Wrong codes to count towards the lockout.
    pub failures: u32,
    /// The marker the transcript given to the model carries, if any.
    pub marker: Option<&'static str>,
}

/// Combine the scans of the whisper transcript and the phone's: a code
/// counts if either authorises it. While locked out nothing is accepted
/// and nothing counts as a failure.
pub fn decide(scans: &[&ScanSummary], locked: bool, origin: i64, valid_for_secs: u64) -> Decision {
    let heard = scans.iter().any(|s| s.heard);
    let contains_code = scans.iter().any(|s| s.contains_code());
    let authorised: BTreeSet<String> = if locked {
        BTreeSet::new()
    } else {
        scans.iter().flat_map(|s| s.authorised()).collect()
    };
    let failures = if locked || !authorised.is_empty() {
        0
    } else {
        scans.iter().map(|s| s.rejected).max().unwrap_or(0)
    };
    let until_t = origin.saturating_add(valid_for_secs as i64);
    let (status, marker) = match (heard, locked, authorised.is_empty()) {
        (false, ..) => (None, None),
        (true, true, _) => (Some("locked_out"), Some(MARKER_LOCKED)),
        (true, false, false) => (Some("accepted"), Some(MARKER_ACCEPTED)),
        (true, false, true) => (Some("rejected"), Some(MARKER_REJECTED)),
    };
    Decision {
        auth: CodeAuth {
            until: authorised.into_iter().map(|c| (c, until_t)).collect(),
            contains_code,
            status: status.map(str::to_string),
        },
        failures,
        marker,
    }
}

/// `text` with every code marker replaced by `marker`, or `marker` added
/// at the end when it has none (the code was heard in the other
/// transcript).
pub fn set_marker(text: &str, marker: Option<&str>) -> String {
    let Some(marker) = marker else {
        return text.to_string();
    };
    let mut out = text.to_string();
    let mut found = false;
    for m in MARKERS {
        if out.contains(m) {
            found = true;
            out = out.replace(m, marker);
        }
    }
    if !found {
        out = format!("{} {marker}", out.trim_end());
    }
    out
}

struct Code {
    name: String,
    /// One tag per spoken variant, and its length in tokens.
    variants: Vec<(usize, Vec<u8>)>,
}

/// The configured codes, ready to check transcripts. Holds no plain code
/// values: only keyed tags (the key is random per process).
pub struct AuthCodes {
    key: [u8; 32],
    triggers: Vec<Vec<String>>,
    codes: Vec<Code>,
    /// Longest variant, in tokens: how much a rejected attempt removes.
    longest: usize,
    pub valid_for_secs: u64,
    pub max_failures: u32,
    pub failure_window_secs: u64,
    pub lockout_secs: u64,
}

impl std::fmt::Debug for AuthCodes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuthCodes")
            .field(
                "codes",
                &self.codes.iter().map(|c| &c.name).collect::<Vec<_>>(),
            )
            .finish_non_exhaustive()
    }
}

impl AuthCodes {
    /// No codes: transcripts are never scanned or changed.
    pub fn disabled() -> Self {
        Self {
            key: [0; 32],
            triggers: Vec::new(),
            codes: Vec::new(),
            longest: 0,
            valid_for_secs: 0,
            max_failures: 0,
            failure_window_secs: 0,
            lockout_secs: 0,
        }
    }

    /// Build from the config, reading each code's value with `lookup`
    /// (`config::secret_from_env` in the service). A code whose variable is
    /// unset is an error: its `code` rules could never pass. Every value is
    /// also registered with `config::redact_credentials`.
    pub fn load(
        cfg: &AuthCodesConfig,
        lookup: impl Fn(&str) -> Result<Option<String>>,
    ) -> Result<Self> {
        if cfg.codes.is_empty() {
            return Ok(Self::disabled());
        }
        let key: [u8; 32] = rand::random();
        let mut triggers: Vec<Vec<String>> = cfg
            .triggers
            .iter()
            .map(|t| normalise(t))
            .filter(|t| !t.is_empty())
            .collect();
        // Longest first: "authorisation code" before "code".
        triggers.sort_by_key(|t| std::cmp::Reverse(t.len()));
        triggers.dedup();
        let mut codes = Vec::new();
        let mut longest = 0;
        for (name, c) in &cfg.codes {
            let Some(value) = lookup(&c.code_env)? else {
                bail!(
                    "{} is not set (needed for auth_codes.codes.{name})",
                    c.code_env
                );
            };
            let mut variants = Vec::new();
            for raw in value.split(['|', '\n']) {
                let tokens = normalise(raw);
                if tokens.is_empty() {
                    continue;
                }
                crate::config::register_secret(raw.trim());
                crate::config::register_secret(&tokens.join(" "));
                if triggers.contains(&tokens) {
                    bail!("a spoken form of auth_codes.codes.{name} is also a trigger phrase");
                }
                longest = longest.max(tokens.len());
                variants.push((tokens.len(), tag(&key, &tokens.join(" "))));
            }
            if variants.is_empty() {
                bail!(
                    "{} holds no words (needed for auth_codes.codes.{name})",
                    c.code_env
                );
            }
            if variants.len() > MAX_VARIANTS {
                bail!(
                    "{} holds {} spoken forms; at most {MAX_VARIANTS} are allowed",
                    c.code_env,
                    variants.len()
                );
            }
            codes.push(Code {
                name: name.clone(),
                variants,
            });
        }
        Ok(Self {
            key,
            triggers,
            codes,
            longest,
            valid_for_secs: cfg.valid_for_secs,
            max_failures: cfg.max_failures,
            failure_window_secs: cfg.failure_window_secs,
            lockout_secs: cfg.lockout_secs,
        })
    }

    /// Whether any code is configured (and transcripts are scanned).
    pub fn enabled(&self) -> bool {
        !self.codes.is_empty()
    }

    /// Codes (by name) that `tokens[at..]` starts with, and the longest
    /// match's length. Every variant is compared, in constant time each.
    fn codes_at(&self, tokens: &[Token], at: usize) -> (BTreeSet<String>, usize) {
        let mut names = BTreeSet::new();
        let mut len = 0;
        for code in &self.codes {
            for (n, want) in &code.variants {
                let Some(window) = tokens.get(at..at + n) else {
                    continue;
                };
                let text = window
                    .iter()
                    .map(|t| t.text.as_str())
                    .collect::<Vec<_>>()
                    .join(" ");
                let mut mac =
                    HmacSha256::new_from_slice(&self.key).expect("HMAC accepts any key length");
                mac.update(text.as_bytes());
                if mac.verify_slice(want).is_ok() {
                    names.insert(code.name.clone());
                    len = len.max(*n);
                }
            }
        }
        (names, len)
    }

    fn trigger_at(&self, tokens: &[Token], at: usize) -> Option<usize> {
        self.triggers
            .iter()
            .find(|t| {
                tokens
                    .get(at..at + t.len())
                    .is_some_and(|w| w.iter().zip(t.iter()).all(|(a, b)| a.text == *b))
            })
            .map(Vec::len)
    }

    /// Find code phrases in `text` and replace them: a trigger phrase and
    /// the code after it by [`MARKER_ACCEPTED`], a trigger phrase and the
    /// words after it (as many as the longest code) by
    /// [`MARKER_REJECTED`], and code words without a trigger phrase by
    /// [`MARKER_REDACTED`]. With no codes configured, `text` is returned
    /// unchanged.
    pub fn scan(&self, text: &str) -> Scan {
        self.scan_with(text, true)
    }

    fn scan_with(&self, text: &str, triggers: bool) -> Scan {
        let mut summary = ScanSummary::default();
        if !self.enabled() {
            return Scan {
                text: text.to_string(),
                summary,
            };
        }
        let (words, tokens) = tokenise(text);
        // (first word, last word, replacement)
        let mut cuts: Vec<(usize, usize, &'static str)> = Vec::new();
        let mut i = 0;
        while i < tokens.len() {
            if let Some(n) = self.trigger_at(&tokens, i).filter(|_| triggers) {
                summary.heard = true;
                let after = i + n;
                let (names, len) = self.codes_at(&tokens, after);
                let (end, marker) = if !names.is_empty() {
                    summary.matched.extend(names);
                    (after + len, MARKER_ACCEPTED)
                } else {
                    if after < tokens.len() {
                        summary.rejected += 1;
                    }
                    // The words that were meant as the code, up to the
                    // longest code's length or the next trigger phrase.
                    let mut end = after;
                    while end < tokens.len()
                        && end < after + self.longest
                        && self.trigger_at(&tokens, end).is_none()
                    {
                        end += 1;
                    }
                    (end, MARKER_REJECTED)
                };
                cuts.push((tokens[i].word, tokens[end - 1].word, marker));
                i = end;
                // The rest of a word already cut (a digit run) goes too.
                while i < tokens.len() && tokens[i].word == tokens[end - 1].word {
                    i += 1;
                }
                continue;
            }
            let (names, len) = self.codes_at(&tokens, i);
            if !names.is_empty() {
                summary.redacted = true;
                let end = i + len;
                cuts.push((tokens[i].word, tokens[end - 1].word, MARKER_REDACTED));
                i = end;
                while i < tokens.len() && tokens[i].word == tokens[end - 1].word {
                    i += 1;
                }
                continue;
            }
            i += 1;
        }
        let mut out = String::with_capacity(text.len());
        let mut pos = 0;
        for (first, last, marker) in cuts {
            out.push_str(&text[pos..words[first].0]);
            out.push_str(marker);
            pos = words[last].1;
        }
        out.push_str(&text[pos..]);
        Scan { text: out, summary }
    }

    /// `text` with any code words removed, trigger phrases left alone
    /// (defence in depth for text that should never hold a code:
    /// notifications, confirmation summaries).
    pub fn scrub(&self, text: &str) -> String {
        if !self.enabled() {
            return text.to_string();
        }
        self.scan_with(text, false).text
    }
}

struct Token {
    text: String,
    /// Index into the word spans.
    word: usize,
}

/// Words as byte spans, and their normalised tokens.
fn tokenise(text: &str) -> (Vec<(usize, usize)>, Vec<Token>) {
    let mut words = Vec::new();
    let mut start = None;
    for (i, c) in text.char_indices() {
        match (c.is_alphanumeric(), start) {
            (true, None) => start = Some(i),
            (false, Some(s)) => {
                words.push((s, i));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        words.push((s, text.len()));
    }
    let mut tokens = Vec::new();
    for (w, &(s, e)) in words.iter().enumerate() {
        for t in normalise_word(&text[s..e]) {
            tokens.push(Token { text: t, word: w });
        }
    }
    (words, tokens)
}

/// One word as tokens: lowercase, number words as digits, digit runs split
/// into single digits ("55" and "five five" compare equal).
fn normalise_word(word: &str) -> Vec<String> {
    let lower = word.to_lowercase();
    if !lower.is_empty() && lower.chars().all(|c| c.is_ascii_digit()) {
        return lower.chars().map(|c| c.to_string()).collect();
    }
    let digit = match lower.as_str() {
        "zero" | "oh" | "nought" => "0",
        "one" => "1",
        "two" => "2",
        "three" => "3",
        "four" => "4",
        "five" => "5",
        "six" => "6",
        "seven" => "7",
        "eight" => "8",
        "nine" | "niner" => "9",
        _ => return vec![lower],
    };
    vec![digit.to_string()]
}

/// A phrase as normalised tokens (punctuation dropped, whitespace
/// collapsed).
pub fn normalise(text: &str) -> Vec<String> {
    tokenise(text).1.into_iter().map(|t| t.text).collect()
}

fn tag(key: &[u8], text: &str) -> Vec<u8> {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC accepts any key length");
    mac.update(text.as_bytes());
    mac.finalize().into_bytes().to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{AuthCodeConfig, AuthCodesConfig};

    pub fn codes(values: &[(&str, &str)]) -> AuthCodes {
        let mut cfg = AuthCodesConfig::default();
        for (name, _) in values {
            cfg.codes.insert(
                name.to_string(),
                AuthCodeConfig {
                    code_env: format!("CODE_{name}"),
                },
            );
        }
        let values: BTreeMap<String, String> = values
            .iter()
            .map(|(n, v)| (format!("CODE_{n}"), v.to_string()))
            .collect();
        AuthCodes::load(&cfg, |var| Ok(values.get(var).cloned())).unwrap()
    }

    #[test]
    fn normalisation() {
        assert_eq!(normalise("Delta, Beta 5."), ["delta", "beta", "5"]);
        assert_eq!(normalise("  delta-beta   FIVE "), ["delta", "beta", "5"]);
        assert_eq!(normalise("zero oh 12"), ["0", "0", "1", "2"]);
        assert!(normalise(" ,.! ").is_empty());
    }

    #[test]
    fn spoken_forms_of_the_code_are_accepted_and_removed() {
        let c = codes(&[("alarm", "Delta Beta 5")]);
        for (said, rest) in [
            (
                "Disarm the house alarm, authorisation code Delta Beta 5",
                "Disarm the house alarm, [authorisation code accepted]",
            ),
            (
                "Disarm the house alarm, authorisation code delta, beta, five.",
                "Disarm the house alarm, [authorisation code accepted].",
            ),
            (
                "Authorization code delta beta five, disarm the alarm",
                "[authorisation code accepted], disarm the alarm",
            ),
            (
                "disarm the alarm auth code DELTA BETA 5",
                "disarm the alarm [authorisation code accepted]",
            ),
            (
                "disarm the alarm code delta beta five",
                "disarm the alarm [authorisation code accepted]",
            ),
        ] {
            let s = c.scan(said);
            assert_eq!(s.text, rest, "{said}");
            assert!(s.summary.heard);
            assert_eq!(
                s.summary.authorised(),
                BTreeSet::from(["alarm".to_string()])
            );
            assert!(!s.text.to_lowercase().contains("delta"));
        }
    }

    #[test]
    fn wrong_codes_are_rejected_and_removed() {
        let c = codes(&[("alarm", "delta beta 5")]);
        let s = c.scan("Disarm the alarm, authorisation code delta beta 6 please now");
        assert_eq!(
            s.text,
            "Disarm the alarm, [authorisation code rejected] please now"
        );
        assert_eq!(s.summary.rejected, 1);
        assert!(s.summary.authorised().is_empty());
        // A trigger with nothing after it is no attempt.
        let s = c.scan("disarm the alarm, authorisation code.");
        assert_eq!(s.text, "disarm the alarm, [authorisation code rejected].");
        assert_eq!(s.summary.rejected, 0);
        assert!(s.summary.heard);
        // Several tries in one recording authorise nothing.
        let s = c.scan("code alpha alpha 1, code delta beta 5");
        assert_eq!(s.summary.matched.len(), 1);
        assert_eq!(s.summary.rejected, 1);
        assert!(s.summary.authorised().is_empty());
    }

    #[test]
    fn variants_but_no_fuzzy_matching() {
        let c = codes(&[("alarm", "delta beta 5 | delta better five")]);
        assert!(
            !c.scan("code delta better 5")
                .summary
                .authorised()
                .is_empty()
        );
        assert!(c.scan("code delta bета 5").summary.authorised().is_empty());
        assert!(c.scan("code delta beat 5").summary.authorised().is_empty());
    }

    #[test]
    fn code_words_without_a_trigger_are_removed_but_authorise_nothing() {
        let c = codes(&[("alarm", "delta beta 5")]);
        let s = c.scan("disarm the alarm, authorization coat delta beta five");
        assert_eq!(s.text, "disarm the alarm, authorization coat [redacted]");
        assert!(s.summary.redacted && !s.summary.heard);
        assert!(s.summary.authorised().is_empty());
        assert!(s.summary.contains_code());
        // Scrubbing removes code words only: "authorisation code" in a
        // reply stays.
        assert_eq!(
            c.scrub("Disarming needs your authorisation code, not Delta Beta 5."),
            "Disarming needs your authorisation code, not [redacted]."
        );
    }

    #[test]
    fn digit_runs_match_spoken_digits() {
        let c = codes(&[("garage", "1 2 3 4")]);
        let s = c.scan("open the garage, code 1234.");
        assert_eq!(s.text, "open the garage, [authorisation code accepted].");
        let s = c.scan("open the garage, code one two three four");
        assert!(s.summary.matched.contains("garage"));
    }

    #[test]
    fn without_codes_nothing_is_touched() {
        let c = AuthCodes::disabled();
        let s = c.scan("add a card for the code review, code delta beta 5");
        assert_eq!(s.text, "add a card for the code review, code delta beta 5");
        assert_eq!(s.summary, ScanSummary::default());
        assert_eq!(c.scrub("code delta beta 5"), "code delta beta 5");
    }

    #[test]
    fn decision_combines_transcripts() {
        let ok = ScanSummary {
            heard: true,
            matched: BTreeSet::from(["alarm".into()]),
            ..Default::default()
        };
        let bad = ScanSummary {
            heard: true,
            rejected: 1,
            ..Default::default()
        };
        let none = ScanSummary::default();
        // Either transcript is enough.
        let d = decide(&[&bad, &ok], false, 1000, 300);
        assert!(d.auth.allows("alarm", 1299) && !d.auth.allows("alarm", 1300));
        assert!(!d.auth.allows("garage", 1000));
        assert_eq!((d.failures, d.marker), (0, Some(MARKER_ACCEPTED)));
        // Both wrong: one failure, not two.
        let d = decide(&[&bad, &bad], false, 1000, 300);
        assert!(d.auth.until.is_empty());
        assert_eq!((d.failures, d.marker), (1, Some(MARKER_REJECTED)));
        // Locked out: nothing accepted, nothing counted.
        let d = decide(&[&ok], true, 1000, 300);
        assert!(d.auth.until.is_empty());
        assert_eq!((d.failures, d.marker), (0, Some(MARKER_LOCKED)));
        assert!(d.auth.contains_code);
        // Nothing heard.
        let d = decide(&[&none], false, 1000, 300);
        assert_eq!(d.marker, None);
        assert!(!d.auth.contains_code);
    }

    #[test]
    fn markers_are_unified() {
        assert_eq!(
            set_marker("x [authorisation code rejected]", Some(MARKER_ACCEPTED)),
            "x [authorisation code accepted]"
        );
        assert_eq!(
            set_marker("disarm the alarm. ", Some(MARKER_ACCEPTED)),
            "disarm the alarm. [authorisation code accepted]"
        );
        assert_eq!(set_marker("x", None), "x");
    }

    #[test]
    fn missing_or_empty_values_refuse_to_load() {
        let mut cfg = AuthCodesConfig::default();
        cfg.codes.insert(
            "alarm".into(),
            AuthCodeConfig {
                code_env: "NOPE".into(),
            },
        );
        let e = AuthCodes::load(&cfg, |_| Ok(None)).unwrap_err();
        assert!(e.to_string().contains("NOPE is not set"), "{e}");
        let e = AuthCodes::load(&cfg, |_| Ok(Some(" , ".into()))).unwrap_err();
        assert!(e.to_string().contains("no words"), "{e}");
        let e = AuthCodes::load(&cfg, |_| Ok(Some("auth code".into()))).unwrap_err();
        assert!(e.to_string().contains("trigger"), "{e}");
        assert!(!format!("{:?}", codes(&[("alarm", "delta beta 5")])).contains("delta"));
    }
}

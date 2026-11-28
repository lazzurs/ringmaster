//! Small statistics for the eval report: AUC, the safest auto-approve
//! cut-off, medians and percentiles, and the title/transcript overlap.

use std::collections::BTreeSet;

/// Area under the ROC curve for scores where higher means "more likely
/// unsafe": the chance that a random unsafe item scores above a random safe
/// one (ties count half). `None` without items of both kinds.
pub fn auc(safe: &[f64], unsafe_scores: &[f64]) -> Option<f64> {
    if safe.is_empty() || unsafe_scores.is_empty() {
        return None;
    }
    let mut wins = 0.0;
    for u in unsafe_scores {
        for s in safe {
            if u > s {
                wins += 1.0;
            } else if u == s {
                wins += 0.5;
            }
        }
    }
    Some(wins / (safe.len() * unsafe_scores.len()) as f64)
}

/// The highest `auto_approve_below` that approves no unsafe item (a call
/// runs when its score is strictly below the cut-off), capped at
/// `threshold` (`check-config` requires `auto_approve_below <= threshold`).
#[derive(Debug, Clone, PartialEq)]
pub struct Cutoff {
    pub cutoff: f64,
    /// Safe items scoring below the cut-off (they would run without asking).
    pub safe_passed: usize,
    pub safe_total: usize,
    /// The cut-off minus the highest safe score it lets through: how far
    /// that safe call is from the nearest unsafe score. `None` when no safe
    /// item passes.
    pub margin: Option<f64>,
    /// The lowest unsafe score (the cut-off before capping).
    pub lowest_unsafe: Option<f64>,
}

pub fn best_cutoff(safe: &[f64], unsafe_scores: &[f64], threshold: f64) -> Cutoff {
    let lowest_unsafe = unsafe_scores.iter().copied().reduce(f64::min);
    let cutoff = lowest_unsafe.unwrap_or(threshold).min(threshold);
    let passed: Vec<f64> = safe.iter().copied().filter(|s| *s < cutoff).collect();
    let margin = passed
        .iter()
        .copied()
        .reduce(f64::max)
        .map(|top| cutoff - top);
    Cutoff {
        cutoff,
        safe_passed: passed.len(),
        safe_total: safe.len(),
        margin,
        lowest_unsafe,
    }
}

/// Median (`None` when empty).
pub fn median(values: &[f64]) -> Option<f64> {
    percentile(values, 50.0)
}

/// Nearest-rank percentile (`None` when empty).
pub fn percentile(values: &[f64], p: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    let rank = ((p / 100.0) * v.len() as f64).ceil() as usize;
    Some(v[rank.clamp(1, v.len()) - 1])
}

const STOPWORDS: &[&str] = &[
    "a", "an", "the", "to", "of", "for", "and", "or", "on", "in", "at", "by", "with", "my", "me",
    "i", "it", "is", "be", "about", "that", "this", "we", "our", "us", "so", "up", "do", "some",
    "remember", "add", "note", "please", "need", "should",
];

/// Content words: lowercase alphanumeric runs, without stopwords and
/// one-letter words, with a plural `s` dropped.
pub fn tokens(text: &str) -> BTreeSet<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() > 1 && !STOPWORDS.contains(w))
        .map(|w| {
            if w.len() > 3 && w.ends_with('s') && !w.ends_with("ss") {
                w[..w.len() - 1].to_string()
            } else {
                w.to_string()
            }
        })
        .collect()
}

/// Share of the title's content words that are in the transcript (0 when
/// the title has none).
pub fn title_overlap(title: &str, transcript: &str) -> f64 {
    let t = tokens(title);
    if t.is_empty() {
        return 0.0;
    }
    let said = tokens(transcript);
    t.iter().filter(|w| said.contains(*w)).count() as f64 / t.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auc_counts_pairs_and_ties() {
        assert_eq!(auc(&[0.1, 0.2], &[0.8, 0.9]), Some(1.0));
        assert_eq!(auc(&[0.8, 0.9], &[0.1, 0.2]), Some(0.0));
        assert_eq!(auc(&[0.5], &[0.5]), Some(0.5));
        // 3 of 4 pairs ordered correctly.
        assert_eq!(auc(&[0.1, 0.6], &[0.5, 0.9]), Some(0.75));
        assert_eq!(auc(&[], &[0.5]), None);
    }

    #[test]
    fn cutoff_approves_no_unsafe_item() {
        let c = best_cutoff(&[0.01, 0.02, 0.30], &[0.05, 0.9], 0.5);
        assert_eq!(c.cutoff, 0.05);
        assert_eq!(c.safe_passed, 2);
        assert_eq!(c.safe_total, 3);
        assert!((c.margin.unwrap() - 0.03).abs() < 1e-12);
        assert_eq!(c.lowest_unsafe, Some(0.05));
        // Capped at the threshold.
        let c = best_cutoff(&[0.1, 0.6], &[0.95], 0.5);
        assert_eq!(c.cutoff, 0.5);
        assert_eq!(c.safe_passed, 1);
        // An unsafe item below every safe one: nothing can be approved.
        let c = best_cutoff(&[0.2, 0.3], &[0.01], 0.5);
        assert_eq!(c.safe_passed, 0);
        assert_eq!(c.margin, None);
        // A tie with an unsafe score does not pass (strictly below).
        let c = best_cutoff(&[0.2], &[0.2], 0.5);
        assert_eq!(c.safe_passed, 0);
    }

    #[test]
    fn percentiles() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), Some(2.0));
        assert_eq!(
            percentile(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 90.0),
            Some(9.0)
        );
        assert_eq!(median(&[]), None);
    }

    #[test]
    fn overlap_of_title_and_transcript() {
        assert_eq!(
            title_overlap(
                "Buy batteries for smoke alarm",
                "buy batteries for the smoke alarm"
            ),
            1.0
        );
        assert_eq!(
            title_overlap("Call plumber", "remember to call the plumber about the tap"),
            1.0
        );
        assert_eq!(title_overlap("Groceries", "buy batteries"), 0.0);
        assert_eq!(title_overlap("Batteries, kite", "buy batteries"), 0.5);
        assert_eq!(title_overlap("", "anything"), 0.0);
    }
}

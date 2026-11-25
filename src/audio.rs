//! M4A (AAC-LC) to 16 kHz mono 16-bit WAV, in pure Rust.
//!
//! The ring app sends AAC-LC in an MP4 container, mono, 16 kHz. whisper.cpp's
//! server only reads WAV (without `--convert` and ffmpeg), so the worker
//! converts before uploading. Other rates or channel counts are downmixed and
//! resampled (linear) to what whisper expects.
//!
//! Decoding runs on a blocking thread, which cannot be cancelled, so it is
//! bounded from inside: [`Budget`] caps the input size and the decoded
//! duration, and the decode loop checks a deadline between packets and
//! returns early. [`Decoder`] adds a semaphore (`stt.decode_concurrency`):
//! a decode that still never returns keeps its permit, so repeated hangs
//! cannot tie up more threads than that; later recordings wait for a permit
//! only until their own deadline, then fall back to the phone's
//! transcription.

use std::io::Cursor;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error as SymError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

pub const WHISPER_RATE: u32 = 16_000;

/// Decoded mono PCM.
#[derive(Debug, Clone)]
pub struct Pcm {
    pub sample_rate: u32,
    pub samples: Vec<f32>,
}

impl Pcm {
    pub fn duration_secs(&self) -> f64 {
        self.samples.len() as f64 / self.sample_rate as f64
    }
}

/// Limits for one decode.
#[derive(Debug, Clone, Copy)]
pub struct Budget {
    /// Stop once this much audio has been decoded.
    pub max_secs: u64,
    /// Stop (with an error) when this passes.
    pub deadline: Instant,
}

impl Budget {
    /// No practical limit (tests, tools).
    pub fn unlimited() -> Self {
        Self {
            max_secs: u64::MAX / 1_000_000,
            deadline: Instant::now() + Duration::from_secs(365 * 86_400),
        }
    }
}

/// Decode an M4A/AAC file into mono f32 samples at the source rate.
pub fn decode_m4a(bytes: &[u8]) -> Result<Pcm> {
    decode_m4a_within(bytes, &Budget::unlimited())
}

/// [`decode_m4a`], giving up when the budget runs out.
pub fn decode_m4a_within(bytes: &[u8], budget: &Budget) -> Result<Pcm> {
    let mss = MediaSourceStream::new(Box::new(Cursor::new(bytes.to_vec())), Default::default());
    let mut hint = Hint::new();
    hint.with_extension("m4a");
    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .context("probing audio container")?;
    let track = format
        .default_track(TrackType::Audio)
        .context("no audio track")?;
    let params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .context("track has no audio codec parameters")?
        .clone();
    let track_id = track.id;
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(&params, &AudioDecoderOptions::default())
        .context("no decoder for the audio codec")?;

    let mut rate = params.sample_rate.unwrap_or(0);
    let mut mono: Vec<f32> = Vec::new();
    let mut interleaved: Vec<f32> = Vec::new();
    let mut decode_errors = 0usize;
    let max_secs = budget.max_secs as f64;
    loop {
        if Instant::now() >= budget.deadline {
            bail!("decoding ran past its deadline (stt.decode_timeout_secs)");
        }
        if rate > 0 && mono.len() as f64 / rate as f64 > max_secs {
            bail!(
                "recording is longer than {} s (stt.max_audio_secs)",
                budget.max_secs
            );
        }
        let packet = match format.next_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break,
            Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e).context("reading audio packet"),
        };
        if packet.track_id != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(buf) => {
                let channels = buf.spec().channels().count().max(1);
                rate = buf.spec().rate();
                interleaved.resize(buf.samples_interleaved(), 0.0);
                buf.copy_to_slice_interleaved(&mut interleaved);
                for frame in interleaved.chunks_exact(channels) {
                    mono.push(frame.iter().sum::<f32>() / channels as f32);
                }
            }
            Err(SymError::DecodeError(_)) => decode_errors += 1,
            Err(e) => return Err(e).context("decoding audio"),
        }
    }
    if rate == 0 {
        bail!("could not determine sample rate");
    }
    if mono.is_empty() {
        bail!("no audio decoded ({decode_errors} bad packets)");
    }
    Ok(Pcm {
        sample_rate: rate,
        samples: mono,
    })
}

/// Linear resampling; adequate for speech going to whisper.
pub fn resample(pcm: &Pcm, target: u32) -> Pcm {
    if pcm.sample_rate == target || pcm.samples.is_empty() {
        return pcm.clone();
    }
    let ratio = pcm.sample_rate as f64 / target as f64;
    let out_len = ((pcm.samples.len() as f64) / ratio).floor().max(1.0) as usize;
    let last = pcm.samples.len() - 1;
    let samples = (0..out_len)
        .map(|i| {
            let pos = i as f64 * ratio;
            let idx = pos.floor() as usize;
            let frac = (pos - idx as f64) as f32;
            let a = pcm.samples[idx.min(last)];
            let b = pcm.samples[(idx + 1).min(last)];
            a + (b - a) * frac
        })
        .collect();
    Pcm {
        sample_rate: target,
        samples,
    }
}

/// Encode mono PCM as a 16-bit WAV file.
pub fn to_wav(pcm: &Pcm) -> Result<Vec<u8>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: pcm.sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut out = Cursor::new(Vec::with_capacity(44 + pcm.samples.len() * 2));
    {
        let mut w = hound::WavWriter::new(&mut out, spec)?;
        for s in &pcm.samples {
            let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
            w.write_sample(v)?;
        }
        w.finalize()?;
    }
    Ok(out.into_inner())
}

/// The whole conversion the worker needs: M4A bytes to a whisper-ready WAV.
pub fn m4a_to_whisper_wav(bytes: &[u8]) -> Result<(Vec<u8>, f64)> {
    m4a_to_whisper_wav_within(bytes, &Budget::unlimited())
}

/// [`m4a_to_whisper_wav`] within a [`Budget`].
pub fn m4a_to_whisper_wav_within(bytes: &[u8], budget: &Budget) -> Result<(Vec<u8>, f64)> {
    let pcm = decode_m4a_within(bytes, budget)?;
    let pcm = resample(&pcm, WHISPER_RATE);
    let secs = pcm.duration_secs();
    Ok((to_wav(&pcm)?, secs))
}

/// What turns recording bytes into a WAV; replaceable for tests.
pub type DecodeFn = Arc<dyn Fn(&[u8], &Budget) -> Result<(Vec<u8>, f64)> + Send + Sync>;

/// Result of [`Decoder::decode`].
#[derive(Debug)]
pub enum Decoded {
    /// WAV bytes and duration in seconds.
    Wav(Vec<u8>, f64),
    /// This recording cannot be decoded (bad, too big, too long, too slow,
    /// or the decoder panicked on it).
    Failed(String),
    /// Every decoder is still busy (most likely stuck on earlier
    /// recordings); this one was not tried.
    Busy,
}

/// Bounded decoding: size and duration caps, a deadline checked inside the
/// decode loop, and at most `stt.decode_concurrency` blocking threads.
#[derive(Clone)]
pub struct Decoder {
    permits: Arc<tokio::sync::Semaphore>,
    max_bytes: u64,
    max_secs: u64,
    timeout: Duration,
    decode: DecodeFn,
    tracker: crate::health::BlockingTracker,
}

impl Decoder {
    pub fn new(cfg: &crate::config::SttConfig, tracker: crate::health::BlockingTracker) -> Self {
        Self::with_fn(cfg, tracker, Arc::new(m4a_to_whisper_wav_within))
    }

    /// A decoder running `decode` instead of the real one.
    pub fn with_fn(
        cfg: &crate::config::SttConfig,
        tracker: crate::health::BlockingTracker,
        decode: DecodeFn,
    ) -> Self {
        Self {
            permits: Arc::new(tokio::sync::Semaphore::new(cfg.decode_concurrency.max(1))),
            max_bytes: cfg.max_audio_bytes,
            max_secs: cfg.max_audio_secs,
            timeout: Duration::from_secs(cfg.decode_timeout_secs),
            decode,
            tracker,
        }
    }

    pub fn max_bytes(&self) -> u64 {
        self.max_bytes
    }

    /// Decode within `stt.decode_timeout_secs`, waiting for a free decoder
    /// no longer than that either. Never waits on a thread past the deadline.
    pub async fn decode(&self, bytes: Vec<u8>) -> Decoded {
        if bytes.len() as u64 > self.max_bytes {
            return Decoded::Failed(format!(
                "recording is {} bytes, over stt.max_audio_bytes ({})",
                bytes.len(),
                self.max_bytes
            ));
        }
        let deadline = Instant::now() + self.timeout;
        let permit =
            match tokio::time::timeout_at(deadline.into(), self.permits.clone().acquire_owned())
                .await
            {
                Ok(Ok(p)) => p,
                _ => return Decoded::Busy,
            };
        let budget = Budget {
            max_secs: self.max_secs,
            deadline,
        };
        let guard = self.tracker.start("decode", self.timeout.as_secs());
        let decode = self.decode.clone();
        let task = tokio::task::spawn_blocking(move || {
            // Held until the thread really returns, however long that is.
            let _permit = permit;
            let _guard = guard;
            decode(&bytes, &budget)
        });
        // The loop checks the deadline between packets; allow it a moment
        // to notice before giving up on the thread.
        let grace = Duration::from_secs(1);
        match tokio::time::timeout_at((deadline + grace).into(), task).await {
            Ok(Ok(Ok((wav, secs)))) => Decoded::Wav(wav, secs),
            Ok(Ok(Err(e))) => Decoded::Failed(format!("{e:#}")),
            Ok(Err(e)) => Decoded::Failed(format!("decoder task: {e}")),
            Err(_) => Decoded::Failed(format!(
                "no result within {} s (stt.decode_timeout_secs)",
                self.timeout.as_secs()
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TONE: &[u8] = include_bytes!("../tests/fixtures/tone-1s.m4a");

    #[test]
    fn decodes_fixture_to_16k_mono_wav() {
        let pcm = decode_m4a(TONE).expect("symphonia decodes AAC-LC in M4A");
        assert_eq!(pcm.sample_rate, 16_000);
        // One second of audio, give or take encoder priming/padding frames.
        let n = pcm.samples.len();
        assert!(
            (15_000..=18_000).contains(&n),
            "unexpected sample count {n}"
        );
        let rms = (pcm.samples.iter().map(|s| s * s).sum::<f32>() / n as f32).sqrt();
        assert!(rms > 0.05, "decoded audio is silent (rms {rms})");

        let (wav, secs) = m4a_to_whisper_wav(TONE).unwrap();
        assert!((0.9..1.2).contains(&secs));
        let reader = hound::WavReader::new(Cursor::new(wav)).unwrap();
        let spec = reader.spec();
        assert_eq!(spec.channels, 1);
        assert_eq!(spec.sample_rate, 16_000);
        assert_eq!(spec.bits_per_sample, 16);
        assert_eq!(reader.len() as usize, n);
    }

    #[test]
    fn decoding_stops_at_the_deadline_and_the_duration_cap() {
        let past = Budget {
            max_secs: 600,
            deadline: Instant::now(),
        };
        let err = decode_m4a_within(TONE, &past).unwrap_err().to_string();
        assert!(err.contains("deadline"), "{err}");
        let short = Budget {
            max_secs: 0,
            deadline: Instant::now() + Duration::from_secs(60),
        };
        let err = decode_m4a_within(TONE, &short).unwrap_err().to_string();
        assert!(err.contains("longer than 0 s"), "{err}");
    }

    fn stt(concurrency: usize, timeout: u64) -> crate::config::SttConfig {
        crate::config::SttConfig {
            decode_concurrency: concurrency,
            decode_timeout_secs: timeout,
            max_audio_bytes: 1000,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn oversized_input_is_not_decoded() {
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let c = calls.clone();
        let d = Decoder::with_fn(
            &stt(1, 5),
            Default::default(),
            Arc::new(move |_: &[u8], _: &Budget| {
                c.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok((vec![], 0.0))
            }),
        );
        match d.decode(vec![0; 1001]).await {
            Decoded::Failed(e) => assert!(e.contains("stt.max_audio_bytes"), "{e}"),
            other => panic!("{other:?}"),
        }
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_decoder_that_never_returns_ties_up_at_most_the_cap() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        // Ignores its budget, like a hang inside the codec would.
        let live = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let calls = Arc::new(AtomicUsize::new(0));
        let (l, p, c) = (live.clone(), peak.clone(), calls.clone());
        let tracker = crate::health::BlockingTracker::default();
        let d = Decoder::with_fn(
            &stt(1, 1),
            tracker.clone(),
            Arc::new(move |_: &[u8], _: &Budget| {
                let n = l.fetch_add(1, Ordering::SeqCst) + 1;
                p.fetch_max(n, Ordering::SeqCst);
                if c.fetch_add(1, Ordering::SeqCst) == 0 {
                    std::thread::sleep(Duration::from_secs(6));
                }
                l.fetch_sub(1, Ordering::SeqCst);
                Ok((vec![], 0.0))
            }),
        );
        let started = Instant::now();
        let mut outcomes = Vec::new();
        for _ in 0..3 {
            let t = Instant::now();
            outcomes.push(d.decode(vec![1; 10]).await);
            assert!(
                t.elapsed() < Duration::from_millis(2500),
                "{:?}",
                t.elapsed()
            );
        }
        assert!(
            matches!(&outcomes[0], Decoded::Failed(e) if e.contains("no result within 1 s")),
            "{outcomes:?}"
        );
        assert!(
            outcomes[1..].iter().all(|o| matches!(o, Decoded::Busy)),
            "{outcomes:?}"
        );
        assert_eq!(
            peak.load(Ordering::SeqCst),
            1,
            "never more threads than the cap"
        );
        assert_eq!(tracker.count("decode"), 1, "the stuck decode is visible");
        assert!(started.elapsed() < Duration::from_millis(5500));
        // Once it returns, the permit and the tracker entry are released.
        tokio::time::sleep(Duration::from_secs(3)).await;
        assert_eq!(tracker.count("decode"), 0);
        assert!(matches!(d.decode(vec![1; 10]).await, Decoded::Wav(..)));
        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "busy attempts never started a thread"
        );
    }

    #[test]
    fn resamples_and_rejects_garbage() {
        let pcm = Pcm {
            sample_rate: 48_000,
            samples: vec![0.5; 48_000],
        };
        let out = resample(&pcm, 16_000);
        assert_eq!(out.samples.len(), 16_000);
        assert!(decode_m4a(b"definitely not audio").is_err());
    }
}

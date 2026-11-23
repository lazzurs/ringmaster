//! M4A (AAC-LC) to 16 kHz mono 16-bit WAV, in pure Rust.
//!
//! The ring app sends AAC-LC in an MP4 container, mono, 16 kHz. whisper.cpp's
//! server only reads WAV (without `--convert` and ffmpeg), so the worker
//! converts before uploading. Other rates or channel counts are downmixed and
//! resampled (linear) to what whisper expects.

use std::io::Cursor;

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

/// Decode an M4A/AAC file into mono f32 samples at the source rate.
pub fn decode_m4a(bytes: &[u8]) -> Result<Pcm> {
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
    loop {
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
    let pcm = decode_m4a(bytes)?;
    let pcm = resample(&pcm, WHISPER_RATE);
    let secs = pcm.duration_secs();
    Ok((to_wav(&pcm)?, secs))
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

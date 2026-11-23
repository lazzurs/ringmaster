//! Webhook signature verification (Index 01 webhooks, version 1).
//!
//! Signed bytes = UTF-8 of
//! `"v1\n" + timestamp + "\n" + deliveryId + "\n" + trigger + "\n" + ("1"|"0") + "\n"`
//! followed by the raw multipart body. The signature is the lowercase hex
//! HMAC-SHA256 of those bytes keyed with the secret's UTF-8 bytes.

use hmac::{Hmac, KeyInit, Mac};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

pub const SUPPORTED_VERSION: &str = "1";

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SignatureError {
    #[error("missing X-Index-Signature or X-Index-Timestamp")]
    Missing,
    #[error("malformed X-Index-Timestamp")]
    BadTimestamp,
    #[error("timestamp outside the allowed window")]
    Expired,
    #[error("malformed signature")]
    Malformed,
    #[error("signature mismatch")]
    Mismatch,
}

/// The headers that take part in the signature.
pub struct SignedHeaders<'a> {
    pub version: &'a str,
    pub timestamp: Option<&'a str>,
    pub signature: Option<&'a str>,
    pub delivery_id: &'a str,
    pub trigger: &'a str,
    pub test: bool,
}

fn mac_for(
    secret: &[u8],
    version: &str,
    timestamp: &str,
    delivery_id: &str,
    trigger: &str,
    test: bool,
    body: &[u8],
) -> HmacSha256 {
    let mut mac = HmacSha256::new_from_slice(secret).expect("HMAC accepts any key length");
    let prefix = format!(
        "v{version}\n{timestamp}\n{delivery_id}\n{trigger}\n{}\n",
        if test { "1" } else { "0" }
    );
    mac.update(prefix.as_bytes());
    mac.update(body);
    mac
}

/// Lowercase hex signature for the given request parts.
pub fn sign(
    secret: &[u8],
    timestamp: &str,
    delivery_id: &str,
    trigger: &str,
    test: bool,
    body: &[u8],
) -> String {
    let mac = mac_for(
        secret,
        SUPPORTED_VERSION,
        timestamp,
        delivery_id,
        trigger,
        test,
        body,
    );
    hex::encode(mac.finalize().into_bytes())
}

/// Verify a request. `now` is unix seconds.
pub fn verify(
    secret: &[u8],
    h: &SignedHeaders<'_>,
    body: &[u8],
    now: i64,
    tolerance_secs: i64,
) -> Result<(), SignatureError> {
    let (Some(ts), Some(sig)) = (h.timestamp, h.signature) else {
        return Err(SignatureError::Missing);
    };
    let ts_num: i64 = ts
        .trim()
        .parse()
        .map_err(|_| SignatureError::BadTimestamp)?;
    if (now - ts_num).abs() > tolerance_secs {
        return Err(SignatureError::Expired);
    }
    let sig = sig.trim();
    if sig.len() != 64 {
        return Err(SignatureError::Malformed);
    }
    let provided = hex::decode(sig.to_ascii_lowercase()).map_err(|_| SignatureError::Malformed)?;
    let mac = mac_for(
        secret,
        h.version,
        ts,
        h.delivery_id,
        h.trigger,
        h.test,
        body,
    );
    // `verify_slice` compares in constant time.
    mac.verify_slice(&provided)
        .map_err(|_| SignatureError::Mismatch)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SECRET: &str = "test-secret-0123456789abcdef0123456789abcdef";
    const BODY: &[u8] =
        b"--b\r\nContent-Disposition: form-data; name=\"client\"\r\n\r\nring\r\n--b--\r\n";

    // Computed independently with Python:
    //   import hmac, hashlib
    //   msg = b"v1\n1700000000\nrec-123\ndouble-click-hold\n0\n" + BODY
    //   hmac.new(SECRET.encode(), msg, hashlib.sha256).hexdigest()
    const EXPECTED: &str = "5064da53a90dc60ffa7e650d6c93c405c2777c8a2d79209e2de5947c272195c7";

    fn headers<'a>(sig: Option<&'a str>, ts: Option<&'a str>) -> SignedHeaders<'a> {
        SignedHeaders {
            version: "1",
            timestamp: ts,
            signature: sig,
            delivery_id: "rec-123",
            trigger: "double-click-hold",
            test: false,
        }
    }

    #[test]
    fn known_vector() {
        assert_eq!(
            sign(
                SECRET.as_bytes(),
                "1700000000",
                "rec-123",
                "double-click-hold",
                false,
                BODY
            ),
            EXPECTED
        );
        verify(
            SECRET.as_bytes(),
            &headers(Some(EXPECTED), Some("1700000000")),
            BODY,
            1_700_000_100,
            300,
        )
        .unwrap();
    }

    #[test]
    fn uppercase_hex_accepted() {
        let upper = EXPECTED.to_ascii_uppercase();
        verify(
            SECRET.as_bytes(),
            &headers(Some(&upper), Some("1700000000")),
            BODY,
            1_700_000_000,
            300,
        )
        .unwrap();
    }

    #[test]
    fn rejects_tampering() {
        let now = 1_700_000_000;
        let ok = headers(Some(EXPECTED), Some("1700000000"));
        // Body changed.
        assert_eq!(
            verify(SECRET.as_bytes(), &ok, b"other", now, 300),
            Err(SignatureError::Mismatch)
        );
        // Wrong secret.
        assert_eq!(
            verify(b"another-secret", &ok, BODY, now, 300),
            Err(SignatureError::Mismatch)
        );
        // Test flag flipped.
        let mut flipped = headers(Some(EXPECTED), Some("1700000000"));
        flipped.test = true;
        assert_eq!(
            verify(SECRET.as_bytes(), &flipped, BODY, now, 300),
            Err(SignatureError::Mismatch)
        );
        // Trigger changed.
        let mut trig = headers(Some(EXPECTED), Some("1700000000"));
        trig.trigger = "single-click-hold";
        assert_eq!(
            verify(SECRET.as_bytes(), &trig, BODY, now, 300),
            Err(SignatureError::Mismatch)
        );
    }

    #[test]
    fn rejects_stale_and_malformed() {
        assert_eq!(
            verify(
                SECRET.as_bytes(),
                &headers(Some(EXPECTED), Some("1700000000")),
                BODY,
                1_700_000_301,
                300
            ),
            Err(SignatureError::Expired)
        );
        assert_eq!(
            verify(
                SECRET.as_bytes(),
                &headers(Some(EXPECTED), Some("1699999699")),
                BODY,
                1_700_000_000,
                300
            ),
            Err(SignatureError::Expired)
        );
        assert_eq!(
            verify(
                SECRET.as_bytes(),
                &headers(Some(EXPECTED), Some("17e8")),
                BODY,
                0,
                300
            ),
            Err(SignatureError::BadTimestamp)
        );
        assert_eq!(
            verify(
                SECRET.as_bytes(),
                &headers(None, Some("1700000000")),
                BODY,
                0,
                300
            ),
            Err(SignatureError::Missing)
        );
        assert_eq!(
            verify(
                SECRET.as_bytes(),
                &headers(Some("zz"), Some("1700000000")),
                BODY,
                1_700_000_000,
                300
            ),
            Err(SignatureError::Malformed)
        );
    }
}

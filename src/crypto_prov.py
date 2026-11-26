from dataclasses import dataclass
import hashlib
import hmac
import secrets


@dataclass(frozen=True)
class ProvenanceRecord:
    asset_id: str
    model_id: str
    payload_bits: str
    commitment_hex: str


class ProvenanceAuthority:
    """Binds a watermark payload to a secret key and model identity via keyed commitments."""

    def __init__(self, master_secret: bytes | None = None) -> None:
        self._secret = master_secret if master_secret is not None else secrets.token_bytes(32)

    @property
    def public_key_fingerprint(self) -> str:
        """Public identifier derived from the secret key without exposing the secret."""
        return hashlib.sha256(b"pub:" + self._secret).hexdigest()[:16]

    def issue_payload(self, asset_id: str, model_id: str, bit_length: int = 32) -> ProvenanceRecord:
        msg = f"{asset_id}|{model_id}".encode("utf-8")
        digest = hmac.new(self._secret, msg, hashlib.sha256).digest()
        bits = "".join(f"{(digest[i // 8] >> (7 - (i % 8))) & 1}" for i in range(bit_length))
        commitment_hex = hashlib.sha256(
            self.public_key_fingerprint.encode("utf-8") + b"|" + bits.encode("ascii") + b"|" + digest
        ).hexdigest()
        return ProvenanceRecord(
            asset_id=asset_id,
            model_id=model_id,
            payload_bits=bits,
            commitment_hex=commitment_hex,
        )

    def verify_payload(self, record: ProvenanceRecord, extracted_bits: str, max_bit_errors: int = 2) -> dict:
        if len(extracted_bits) != len(record.payload_bits):
            return {"verified": False, "reason": "length_mismatch", "bit_errors": len(record.payload_bits)}

        bit_errors = sum(a != b for a, b in zip(record.payload_bits, extracted_bits))
        expected_msg = f"{record.asset_id}|{record.model_id}".encode("utf-8")
        expected_digest = hmac.new(self._secret, expected_msg, hashlib.sha256).digest()
        expected_commitment = hashlib.sha256(
            self.public_key_fingerprint.encode("utf-8")
            + b"|"
            + record.payload_bits.encode("ascii")
            + b"|"
            + expected_digest
        ).hexdigest()

        commitment_valid = hmac.compare_digest(record.commitment_hex, expected_commitment)
        within_tolerance = bit_errors <= max_bit_errors

        return {
            "verified": commitment_valid and within_tolerance,
            "commitment_valid": commitment_valid,
            "bit_errors": bit_errors,
            "ber": bit_errors / len(record.payload_bits),
            "within_tolerance": within_tolerance,
        }
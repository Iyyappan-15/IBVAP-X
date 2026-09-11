import hashlib
import os
import tempfile
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class EvidenceHasher:
    """Computes and verifies cryptographic SHA-256 tamper-evident hashes for captured evidence files."""

    @staticmethod
    def compute_sha256(filepath: str) -> str:
        """Computes SHA-256 hex digest for a file on disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Evidence file not found: {filepath}")

        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest().upper()

    @staticmethod
    def verify_integrity(filepath: str, expected_hash: str) -> Tuple[bool, str]:
        """
        Recomputes hash of filepath and compares against recorded hash.
        Returns Tuple[is_valid: bool, computed_hash: str].
        """
        computed = EvidenceHasher.compute_sha256(filepath)
        is_valid = (computed.upper() == expected_hash.upper())
        return is_valid, computed

    @staticmethod
    def simulate_tampering_demo(filepath: str, recorded_hash: str) -> Tuple[bool, str, str]:
        """
        Simulates file tampering for demo presentation without touching original sealed file.
        Creates a temporary copy, flips 1 byte, calculates hash mismatch, and deletes temp file.
        Returns Tuple[is_valid (False), recorded_hash, tampered_hash].
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Evidence file not found: {filepath}")

        # Read original bytes
        with open(filepath, "rb") as f:
            data = bytearray(f.read())

        if len(data) == 0:
            return False, recorded_hash, "EMPTY_FILE"

        # Flip the first byte
        data[0] = (data[0] + 1) % 256

        # Write to temporary demo copy
        with tempfile.NamedTemporaryFile(delete=False, suffix="_tampered_demo.tmp") as temp_f:
            temp_f.write(data)
            temp_path = temp_f.name

        try:
            tampered_hash = EvidenceHasher.compute_sha256(temp_path)
            is_valid = (tampered_hash.upper() == recorded_hash.upper())
            logger.info(f"[EvidenceHasher Demo] Original Hash: {recorded_hash[:10]}..., Tampered Hash: {tampered_hash[:10]}...")
            return is_valid, recorded_hash, tampered_hash
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

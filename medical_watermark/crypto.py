"""AES helpers for watermark bitstream protection."""

from __future__ import annotations

import hashlib

import numpy as np
from Crypto.Cipher import AES


def derive_aes_key(secret: str) -> bytes:
    """Derive a stable 128-bit AES key from a user-facing secret string."""
    return hashlib.sha256(secret.encode("utf-8")).digest()[:16]


def encrypt_bit_payload_aes(bits: np.ndarray, key: bytes, nonce: bytes) -> np.ndarray:
    """
    Encrypt a binary payload with AES-CTR and return encrypted bits in {0, 1}.

    The input length must be a multiple of 8 so it can be packed into bytes.
    """
    b = (np.asarray(bits, dtype=np.float64).ravel() >= 0.5).astype(np.uint8)
    if b.size % 8:
        raise ValueError("AES watermark encryption requires payload length divisible by 8 bits")
    plain = np.packbits(b, bitorder="big").tobytes()
    cipher = AES.new(key, AES.MODE_CTR, nonce=nonce)
    enc = cipher.encrypt(plain)
    return np.unpackbits(np.frombuffer(enc, dtype=np.uint8), bitorder="big").astype(np.float64)


def decrypt_bit_payload_aes(bits: np.ndarray, key: bytes, nonce: bytes) -> np.ndarray:
    """Inverse of :func:`encrypt_bit_payload_aes` for AES-CTR protected payloads."""
    b = (np.asarray(bits, dtype=np.float64).ravel() >= 0.5).astype(np.uint8)
    if b.size % 8:
        raise ValueError("AES watermark decryption requires payload length divisible by 8 bits")
    enc = np.packbits(b, bitorder="big").tobytes()
    cipher = AES.new(key, AES.MODE_CTR, nonce=nonce)
    dec = cipher.decrypt(enc)
    return np.unpackbits(np.frombuffer(dec, dtype=np.uint8), bitorder="big").astype(np.float64)

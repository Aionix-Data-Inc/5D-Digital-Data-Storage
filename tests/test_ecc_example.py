from __future__ import annotations

from optical_storage import Parity8, LaserWriter, LaserReader, bytes_to_bits


def test_parity8_encode_decode() -> None:
    ecc = Parity8()
    payload = b"\xAA"  # 10101010
    bits = bytes_to_bits(payload)
    encoded = ecc.encode(bits)
    result = ecc.decode(encoded)
    # decoded bits should match original payload bits
    assert result.bits[: len(bits)] == bits
    assert result.detected_uncorrectable == 0


def test_roundtrip_with_parity8() -> None:
    payload = b"ParityTest"
    writer = LaserWriter(
        grid_size=(16, 16, 2),
        intensity_levels=8,
        polarization_states=8,
        intensity_range=(0.2, 1.0),
        polarization_range=(0.0, 3.14159),
        error_correction=Parity8(),
    )

    pattern = writer.write(payload)
    reader = LaserReader(pattern)
    result = reader.read()
    assert result.data == payload
*** End Patch
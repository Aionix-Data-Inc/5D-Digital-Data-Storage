from __future__ import annotations

from optical_storage import LaserWriter, StoragePattern


def test_storage_pattern_roundtrip_json() -> None:
    payload = b"SerializeMe"
    writer = LaserWriter(
        grid_size=(8, 8, 1),
        intensity_levels=4,
        polarization_states=4,
        intensity_range=(0.1, 0.9),
        polarization_range=(0.0, 3.14159),
    )

    pattern = writer.write(payload)
    data = pattern.to_dict()
    restored = StoragePattern.from_dict(data)

    # Basic checks
    assert restored.grid_size == pattern.grid_size
    assert restored.voxel_count == pattern.voxel_count
    assert restored.intensity_levels == pattern.intensity_levels
    assert restored.polarization_states == pattern.polarization_states
    assert restored.data_length_bytes == pattern.data_length_bytes
*** End Patch
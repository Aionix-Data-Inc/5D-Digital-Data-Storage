from __future__ import annotations

from host_writer import HostWriter


def main() -> None:
    hw = HostWriter(grid_size=(8, 8, 2), intensity_levels=4, polarization_states=4)
    msg = b"5D host writer demo"
    pattern = hw.write(msg)
    rb = hw.verify(pattern)
    print("voxels:", pattern.voxel_count, "corrected:", rb.read_result.corrected_errors)
    print("roundtrip:", rb.data)


if __name__ == "__main__":
    main()

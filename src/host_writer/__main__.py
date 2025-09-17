from __future__ import annotations

import argparse
import sys

from .orchestrator import HostWriter


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Host writer demo CLI")
    p.add_argument("data", help="Data to store (as a literal string)")
    p.add_argument("--grid", default="8x8x2", help="Grid size as XxYxZ, default 8x8x2")
    p.add_argument("--levels", default="4x4", help="Intensity x Polarization levels, default 4x4")
    p.add_argument("--no-scramble", action="store_true", help="Disable scrambler")
    args = p.parse_args(argv)

    try:
        gx, gy, gz = (int(x) for x in args.grid.lower().split("x"))
        il, pl = (int(x) for x in args.levels.lower().split("x"))
    except Exception as e:  # noqa: BLE001
        print(f"Invalid --grid/--levels: {e}", file=sys.stderr)
        return 2

    hw = HostWriter(
        grid_size=(gx, gy, gz),
        intensity_levels=il,
        polarization_states=pl,
        scramble=not args.no_scramble,
    )

    data = args.data.encode("utf-8", errors="strict")
    pattern = hw.write(data)
    rb = hw.verify(pattern)
    ok = rb.data == data and rb.read_result.detected_uncorrectable == 0
    print("Wrote pattern with", rb.pattern.voxel_count, "voxels; corrected_errors=", rb.read_result.corrected_errors)
    print("Roundtrip:", "OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover - manual entry
    raise SystemExit(main())

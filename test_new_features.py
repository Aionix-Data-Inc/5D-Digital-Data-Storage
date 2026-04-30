#!/usr/bin/env python3
"""Test the new laser instruction generation and compensations."""

from src.optical_storage import LaserWriter, LaserInstruction

def main():
    # Create a writer
    writer = LaserWriter(
        grid_size=(4, 4, 2),
        intensity_levels=4,
        polarization_states=4,
    )

    # Write some data
    data = b"test"
    pattern = writer.write(data)

    print("Original voxels:")
    for voxel in pattern.voxels[:5]:  # First 5
        print(f"  {voxel}")

    # Generate laser instructions
    instructions = writer.generate_laser_instructions(pattern)

    print("\nLaser instructions:")
    for instr in instructions[:5]:
        print(f"  {instr}")

    # Test validation (simulate feedback)
    target = pattern.voxels[0]
    feedback_intensity = target.intensity * 0.95  # 5% error
    feedback_polarization = target.polarization
    valid = writer.validate_voxel_formation(target, feedback_intensity, feedback_polarization, tolerance=0.1)
    print(f"\nValidation result: {valid}")

if __name__ == "__main__":
    main()
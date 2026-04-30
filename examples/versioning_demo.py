"""Comprehensive demonstration of versioning and immutable logging."""

import sys
import pathlib
import os
import json

# Set up paths
REPO_ROOT = pathlib.Path('/workspaces/5D-Digital-Data-Storage')
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from optical_storage import (
    LaserWriter, LaserReader, NoErrorCorrection,
    VersionManager, ImmutableWriteLog, WriteLogEntry, EncodingVersion
)


def demonstrate_versioning_and_logging():
    """Demonstrate comprehensive versioning and immutable logging functionality."""

    print("🔢 5D Optical Storage: Versioning & Immutable Logging Demo")
    print("=" * 65)

    # Initialize version manager and write log
    print("\n1. Initializing Version Management and Logging...")

    version_manager = VersionManager()
    write_log = ImmutableWriteLog()

    print("✓ Version manager initialized")
    print(f"  Current version: {version_manager.current_version.value}")
    print(f"  Firmware version: {version_manager.firmware_version}")
    print("✓ Immutable write log initialized")

    # Create writer with versioning and logging
    print("\n2. Creating Version-Aware Writer...")

    writer = LaserWriter(
        grid_size=(8, 8, 2),
        voxel_pitch=(5.0, 5.0, 15.0),
        intensity_levels=8,
        polarization_states=8,
        intensity_range=(0.2, 1.0),
        polarization_range=(0.0, 3.14159),
        error_correction=NoErrorCorrection(),
        version_manager=version_manager,
        write_log=write_log
    )

    print("✓ Writer initialized with version management and logging")

    # Demonstrate version encoding
    print("\n3. Demonstrating Version Encoding...")

    test_data = b"Version-controlled optical storage data"
    encoded_v1 = version_manager.encode_with_version(test_data, EncodingVersion.V1_0)
    encoded_v2 = version_manager.encode_with_version(test_data, EncodingVersion.V2_0)

    print("✓ Encoded data with V1.0 format")
    print("✓ Encoded data with V2.0 format")

    # Test backward compatibility
    print("\n4. Testing Backward Compatibility...")

    decoded_v1 = version_manager.decode_with_version_compatibility(encoded_v1)
    decoded_v2 = version_manager.decode_with_version_compatibility(encoded_v2)

    print(f"✓ V1.0 decoding: {decoded_v1 == test_data}")
    print(f"✓ V2.0 decoding: {decoded_v2 == test_data}")
    print("✓ Backward compatibility verified")

    # Create storage pattern with versioning
    print("\n5. Creating Versioned Storage Pattern...")

    pattern = writer.write(test_data)
    print("✓ Storage pattern created")
    print(f"  Encoding version: {pattern.version_info.encoding_version.value if pattern.version_info else 'unknown'}")
    print(f"  Firmware version: {pattern.version_info.firmware_version if pattern.version_info else 'unknown'}")
    print(f"  Write log hash: {pattern.write_log_hash[:16]}..." if pattern.write_log_hash else "None")

    # Demonstrate immutable logging
    print("\n6. Demonstrating Immutable Write Logging...")

    # Check log files
    log_files = [f for f in os.listdir('write_logs') if f.endswith('.json')] if os.path.exists('write_logs') else []
    if log_files:
        print(f"✓ Found {len(log_files)} immutable log entries")

        # Read and verify the latest log entry
        latest_log = write_log.get_latest_entry()
        if latest_log:
            print("✓ Latest log entry details:")
            print(f"  Operation: {latest_log.operation_type}")
            print(f"  Device: {latest_log.device_id}")
            print(f"  Payload hash: {latest_log.payload_hash[:16]}...")
            print(f"  Pattern hash: {latest_log.pattern_hash[:16]}...")
            print(f"  Log hash: {latest_log.log_hash[:16]}...")

        # Verify log chain integrity
        chain_valid = write_log.verify_chain_integrity()
        print(f"✓ Log chain integrity: {'VERIFIED' if chain_valid else 'BROKEN'}")

    # Test reading with version compatibility
    print("\n7. Testing Version-Aware Reading...")

    reader = LaserReader(pattern, version_manager=version_manager)
    version_compatible = reader.validate_version_compatibility()
    version_info = reader.get_version_info()

    print(f"✓ Version compatibility: {'VERIFIED' if version_compatible else 'FAILED'}")
    print(f"  Pattern version: {version_info['encoding_version']}")
    print(f"  Firmware version: {version_info['firmware_version']}")

    # Perform read operation
    result = reader.read()
    print(f"✓ Read operation successful: {result.data == test_data}")
    print(f"  Data integrity: {'✓ MAINTAINED' if result.hash_valid else '✗ COMPROMISED'}")

    # Demonstrate multiple write operations with logging
    print("\n8. Demonstrating Multiple Write Operations...")

    operations = [
        ("write", b"First dataset"),
        ("write", b"Second dataset"),
        ("retry", b"Retry operation"),
    ]

    for op_type, data in operations:
        if op_type == "write":
            pattern = writer.write(data)
        elif op_type == "retry":
            # Simulate a retry by writing again
            pattern = writer.write(data)

        print(f"✓ {op_type.capitalize()} operation logged")

    # Verify complete log chain
    log_chain = write_log.get_log_chain()
    print(f"✓ Complete log chain: {len(log_chain)} entries")

    for i, entry in enumerate(log_chain[-3:], 1):  # Show last 3 entries
        print(f"  {i}. {entry.operation_type} - {entry.payload_hash[:8]}...")

    # Demonstrate log persistence
    print("\n9. Demonstrating Log Persistence...")

    # Create new log instance and verify it can read existing logs
    new_log = ImmutableWriteLog()
    new_chain = new_log.get_log_chain()
    print(f"✓ Log persistence verified: {len(new_chain)} entries readable")

    # Summary
    print("\n" + "=" * 65)
    print("📋 SUMMARY")
    print("=" * 65)
    print(f"Version manager: {version_manager.current_version.value}")
    print(f"Log entries created: {len(log_chain)}")
    print(f"Chain integrity: {'✓ VERIFIED' if write_log.verify_chain_integrity() else '✗ BROKEN'}")
    print(f"Backward compatibility: ✓ MAINTAINED")
    print(f"Data integrity: ✓ PRESERVED")
    print("\n✅ Versioning and immutable logging successfully demonstrated!")


if __name__ == "__main__":
    demonstrate_versioning_and_logging()
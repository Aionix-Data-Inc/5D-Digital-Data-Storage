"""Version management and immutable logging for optical storage."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum


class EncodingVersion(Enum):
    """Supported encoding scheme versions."""
    V1_0 = "1.0"  # Initial version
    V1_1 = "1.1"  # Added anomaly detection
    V1_2 = "1.2"  # Added immutable logging
    V2_0 = "2.0"  # Major revision with enhanced error correction


@dataclass
class VersionInfo:
    """Version information for encoding schemes."""
    encoding_version: EncodingVersion
    firmware_version: str
    created_at: datetime = field(default_factory=datetime.now)
    compatible_versions: List[str] = field(default_factory=lambda: ["1.0", "1.1", "1.2"])

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "encoding_version": self.encoding_version.value,
            "firmware_version": self.firmware_version,
            "created_at": self.created_at.isoformat(),
            "compatible_versions": self.compatible_versions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VersionInfo":
        """Deserialize from dictionary."""
        return cls(
            encoding_version=EncodingVersion(data["encoding_version"]),
            firmware_version=data["firmware_version"],
            created_at=datetime.fromisoformat(data["created_at"]),
            compatible_versions=data.get("compatible_versions", ["1.0", "1.1", "1.2"]),
        )

    def is_compatible(self, other_version: str) -> bool:
        """Check if this version is compatible with another version."""
        return other_version == self.encoding_version.value or other_version in self.compatible_versions


@dataclass
class WriteLogEntry:
    """An immutable log entry for write operations."""
    timestamp: datetime
    device_id: str
    media_batch_id: str
    operation_type: str  # "write", "retry", "calibration_update"
    payload_hash: str
    pattern_hash: str
    previous_log_hash: Optional[str] = field(default=None)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def log_hash(self) -> str:
        """Calculate the hash of this log entry."""
        data = {
            "timestamp": self.timestamp.isoformat(),
            "device_id": self.device_id,
            "media_batch_id": self.media_batch_id,
            "operation_type": self.operation_type,
            "payload_hash": self.payload_hash,
            "pattern_hash": self.pattern_hash,
            "previous_log_hash": self.previous_log_hash,
            "metadata": json.dumps(self.metadata, sort_keys=True),
        }
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "device_id": self.device_id,
            "media_batch_id": self.media_batch_id,
            "operation_type": self.operation_type,
            "payload_hash": self.payload_hash,
            "pattern_hash": self.pattern_hash,
            "previous_log_hash": self.previous_log_hash,
            "log_hash": self.log_hash,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WriteLogEntry":
        """Deserialize from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            device_id=data["device_id"],
            media_batch_id=data["media_batch_id"],
            operation_type=data["operation_type"],
            payload_hash=data["payload_hash"],
            pattern_hash=data["pattern_hash"],
            previous_log_hash=data["previous_log_hash"],
            metadata=data.get("metadata", {}),
        )


class ImmutableWriteLog:
    """Immutable append-only log for write operations."""

    def __init__(self, log_dir: str = "write_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self._current_log_hash: Optional[str] = self._load_current_log_hash()
        if self._current_log_hash and not self.verify_chain_integrity():
            self.logger.warning("Immutable write log chain integrity check failed for %s", self.log_dir)

    def _load_entries(self) -> tuple[Dict[str, WriteLogEntry], Dict[Optional[str], List[WriteLogEntry]]]:
        """Load all existing log entries from disk."""
        entries: Dict[str, WriteLogEntry] = {}
        next_by_prev: Dict[Optional[str], List[WriteLogEntry]] = {}

        for log_file in os.listdir(self.log_dir):
            if not log_file.endswith('.json'):
                continue

            filepath = os.path.join(self.log_dir, log_file)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    entry = WriteLogEntry.from_dict(data)
            except (json.JSONDecodeError, KeyError, ValueError):
                continue

            entries[entry.log_hash] = entry
            next_by_prev.setdefault(entry.previous_log_hash, []).append(entry)

        return entries, next_by_prev

    def _select_active_chain(
        self,
        entries: Dict[str, WriteLogEntry],
        next_by_prev: Dict[Optional[str], List[WriteLogEntry]],
    ) -> List[WriteLogEntry]:
        """Select the most recent valid chain from the available log entries."""
        roots = [entry for entry in entries.values() if entry.previous_log_hash is None]
        if not roots:
            return []

        best_chain: List[WriteLogEntry] = []
        for root in roots:
            chain = [root]
            current = root
            while True:
                next_entries = next_by_prev.get(current.log_hash, [])
                if len(next_entries) != 1:
                    break
                current = next_entries[0]
                chain.append(current)

            if (
                not best_chain
                or current.timestamp > best_chain[-1].timestamp
                or (
                    current.timestamp == best_chain[-1].timestamp
                    and len(chain) > len(best_chain)
                )
            ):
                best_chain = chain

        return best_chain

    def _load_current_log_hash(self) -> Optional[str]:
        """Load the latest hash for the active log chain."""
        if not os.path.exists(self.log_dir):
            return None

        entries, next_by_prev = self._load_entries()
        chain = self._select_active_chain(entries, next_by_prev)
        return chain[-1].log_hash if chain else None

    def append_entry(self, entry: WriteLogEntry) -> str:
        """Append a new entry to the immutable log."""
        # Set the previous hash
        entry.previous_log_hash = self._current_log_hash

        # Calculate and verify the entry hash
        entry_hash = entry.log_hash

        # Save to file
        log_file = os.path.join(self.log_dir, f"{entry.timestamp.strftime('%Y%m%d_%H%M%S')}_{entry_hash[:8]}.json")
        with open(log_file, 'w') as f:
            json.dump(entry.to_dict(), f, indent=2)

        # Update current hash
        self._current_log_hash = entry_hash

        return entry_hash

    def get_log_chain(self, start_hash: Optional[str] = None) -> List[WriteLogEntry]:
        """Get the log chain starting from a specific hash."""
        if not os.path.exists(self.log_dir):
            return []

        entries, next_by_prev = self._load_entries()
        if start_hash is not None:
            if start_hash not in entries:
                raise ValueError(f"Start hash not found: {start_hash}")

            chain: List[WriteLogEntry] = [entries[start_hash]]
            current = chain[0]
            while True:
                next_entries = next_by_prev.get(current.log_hash, [])
                if len(next_entries) != 1:
                    break
                current = next_entries[0]
                chain.append(current)
            return chain

        return self._select_active_chain(entries, next_by_prev)

    def verify_chain_integrity(self) -> bool:
        """Verify the entire log chain integrity."""
        if not os.path.exists(self.log_dir):
            return True

        for log_file in os.listdir(self.log_dir):
            if not log_file.endswith('.json'):
                continue

            filepath = os.path.join(self.log_dir, log_file)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    entry = WriteLogEntry.from_dict(data)
            except (json.JSONDecodeError, KeyError, ValueError):
                return False

            if data.get("log_hash") != entry.log_hash:
                return False

        try:
            self.get_log_chain()
            return True
        except ValueError:
            return False

    def get_latest_entry(self) -> Optional[WriteLogEntry]:
        """Get the latest log entry."""
        entries = self.get_log_chain()
        return entries[-1] if entries else None

    def find_entries_by_payload_hash(self, payload_hash: str) -> List[WriteLogEntry]:
        """Find log entries that match a specific payload hash."""
        entries, _ = self._load_entries()
        return [entry for entry in entries.values() if entry.payload_hash == payload_hash]


class VersionManager:
    """Manages encoding scheme versions and backward compatibility."""

    def __init__(self):
        self.current_version = EncodingVersion.V2_0
        self.firmware_version = "2.0.0"
        self.version_registry = {
            EncodingVersion.V1_0: self._decode_v1_0,
            EncodingVersion.V1_1: self._decode_v1_1,
            EncodingVersion.V1_2: self._decode_v1_2,
            EncodingVersion.V2_0: self._decode_v2_0,
        }

    def create_version_info(self) -> VersionInfo:
        """Create version info for current encoding."""
        all_supported_versions = [version.value for version in self.version_registry.keys()]
        return VersionInfo(
            encoding_version=self.current_version,
            firmware_version=self.firmware_version,
            compatible_versions=all_supported_versions
        )

    def encode_with_version(self, data: bytes, version: EncodingVersion = None) -> Dict[str, Any]:
        """Encode data with version information."""
        if version is None:
            version = self.current_version

        version_info = VersionInfo(
            encoding_version=version,
            firmware_version=self.firmware_version
        )

        # For now, return basic structure - actual encoding logic would go here
        return {
            "version_info": version_info.to_dict(),
            "data": data.hex(),  # Placeholder encoding
            "encoding_scheme": version.value,
        }

    def decode_with_version_compatibility(self, encoded_data: Dict[str, Any]) -> bytes:
        """Decode data with automatic version detection and backward compatibility."""
        version_info_dict = encoded_data.get("version_info", {})
        if not version_info_dict:
            # No version info - assume V1.0
            version = EncodingVersion.V1_0
        else:
            version_info = VersionInfo.from_dict(version_info_dict)
            version = version_info.encoding_version

        # Check compatibility
        current_info = self.create_version_info()
        if not current_info.is_compatible(version.value):
            raise ValueError(f"Incompatible version: {version.value}")

        # Use appropriate decoder
        decoder = self.version_registry.get(version)
        if not decoder:
            raise ValueError(f"No decoder available for version: {version.value}")

        return decoder(encoded_data)

    def _decode_v1_0(self, encoded_data: Dict[str, Any]) -> bytes:
        """Decode V1.0 format (basic hex decoding)."""
        return bytes.fromhex(encoded_data["data"])

    def _decode_v1_1(self, encoded_data: Dict[str, Any]) -> bytes:
        """Decode V1.1 format (with anomaly detection metadata)."""
        # V1.1 has additional anomaly metadata but same basic decoding
        return bytes.fromhex(encoded_data["data"])

    def _decode_v1_2(self, encoded_data: Dict[str, Any]) -> bytes:
        """Decode V1.2 format (with immutable logging)."""
        # V1.2 includes log references but same basic decoding
        return bytes.fromhex(encoded_data["data"])

    def _decode_v2_0(self, encoded_data: Dict[str, Any]) -> bytes:
        """Decode V2.0 format (enhanced error correction)."""
        # V2.0 uses improved encoding scheme
        return bytes.fromhex(encoded_data["data"])
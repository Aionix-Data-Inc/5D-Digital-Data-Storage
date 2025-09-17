"""Host-side writer orchestration utilities.

This package provides a minimal orchestration layer on top of the
low-level optical storage primitives in :mod:`optical_storage`.

The goal is to demonstrate how higher-level systems may package data,
optionally scramble it, and drive the :class:`~optical_storage.writer.LaserWriter`
to produce a :class:`~optical_storage.storage_pattern.StoragePattern` that can be
read back via :class:`~optical_storage.reader.LaserReader`.
"""

from .orchestrator import HostWriter, HostReadback

__all__ = ["HostWriter", "HostReadback"]

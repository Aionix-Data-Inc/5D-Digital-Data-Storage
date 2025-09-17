# Changelog

## Unreleased

- Add unit tests: Hamming(7,4) single/double-bit behavior, quantisation rounding consistency, and noise clipping/read stability.
- Introduce `host_writer` orchestrator with optional scrambler and `HostWriter.verify` for readback.
- Add `host-writer` CLI entrypoint and `examples/host_writer_demo.py`.
- CI: run tests on push/PR for Python 3.10–3.12.

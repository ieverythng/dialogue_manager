# Local Delta Notes

- Upstream package remains the canonical baseline for the dialogue runtime.
- The active local runtime in `src/dialogue_manager` currently adds:
  - `/chatbot/*` bridge topics
  - ASR holdoff/de-duplication logic
  - NAO-specific say dispatch and `/speech` mirroring
- These behaviors should be externalized or upstreamed selectively instead of
  replacing the upstream node contract.

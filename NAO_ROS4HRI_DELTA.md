# Local Delta Notes

- `src/dialogue_manager` now runs the upstream ROS4HRI dialogue-manager
  baseline directly.
- The old local NAO bridge runtime has been archived under
  `.migration_backups/dialogue_manager_legacy_bridge_20260313/`.
- The only intentional local delta kept in the active package is a temporary
  executable compatibility alias:
  - `dialogue_manager_node -> dialogue_manager.start_manager:main`
- The migration launch overrides the upstream `chatbot` parameter to
  `chatbot_llm` so the canonical `/skill/chat`, `/skill/ask`, and `/skill/say`
  servers bind to the migrated backend.
- Any future NAO-specific dialogue behavior should stay outside the canonical
  runtime unless it is cleanly upstreamable.

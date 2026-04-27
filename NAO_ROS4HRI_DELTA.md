# Local Delta Notes

- `src/dialogue_manager` now runs the upstream ROS4HRI dialogue-manager
  baseline directly.
- Core upstream runtime files are intentionally left functionally aligned with
  upstream:
  - `dialogue_manager/manager_node.py`
  - `dialogue_manager/chatbot_client.py`
  - `dialogue_manager/dialogue.py`
  - `dialogue_manager/skill_servers.py`
  - `dialogue_manager/speech_handler.py`
  - `dialogue_manager/tts_client.py`
- Local modifications are limited to packaging, launch metadata, documentation,
  and the safe-shutdown wrapper in `dialogue_manager/start_manager.py`.
- The migration launch overrides the upstream `chatbot` parameter to
  `chatbot_llm` so the canonical `/skill/chat`, `/skill/ask`, and `/skill/say`
  servers bind to the migrated backend.
- Any future NAO-specific dialogue behavior should stay outside the canonical
  runtime unless it is cleanly upstreamable.

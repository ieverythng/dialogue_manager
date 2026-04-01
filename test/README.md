# Tests

## Code quality

| File | What it checks |
|------|----------------|
| `test_flake8.py` | PEP 8 style (via ament_flake8) |
| `test_pep257.py` | Docstring conventions (via ament_pep257) |
| `test_copyright.py` | Apache 2.0 license headers on all source files |

## Unit tests

| File | Module under test | What it covers |
|------|-------------------|----------------|
| `test_dialogue.py` | `dialogue.py` | `DialogueState`, `Dialogue` dataclass, `DialogueManager` priority logic |
| `test_tts_client.py` | `tts_client.py` | TTS action client lifecycle, `speak()`, `speak_and_wait()`, callbacks |
| `test_chatbot_client.py` | `chatbot_client.py` | Chatbot interaction, `send_input()`, response handling, default chat |
| `test_speech_handler.py` | `speech_handler.py` | Speech input subscription, chatbot forwarding, intent publishing |
| `test_skill_servers.py` | `skill_servers.py` | Chat/Ask/Say goal acceptance, priority filtering, execution |
| `test_manager_node.py` | `manager_node.py` | Lifecycle callbacks (configure/activate/deactivate/shutdown), parameter declaration |
| `test_markup_parser.py` | `markup/parser.py` | Hand-written parser against all valid/invalid examples, detailed AST structure |
| `test_action_library.py` | `markup/action_library.py` | YAML config loading, `#1$name\|default` field template resolution, variable resolution |
| `test_expression_executor.py` | `markup/executor.py` | Text/variable/pause/action execution, cancellation, plain text extraction |

## Integration tests

| File | What it covers |
|------|----------------|
| `test_integration.py` | End-to-end with mock ASR, chatbot, and TTS nodes on a real ROS 2 executor |

## Test data

| Path | Purpose |
|------|---------|
| `markup_valid_examples` | Lines that the markup parser must accept (one expression per line) |
| `markup_invalid_examples` | Lines that the markup parser must reject |

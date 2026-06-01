# Copyright (c) 2026 IIIA-CSIC. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Integration tests for manager_node.py lifecycle node."""

from dialogue_manager.manager_node import DialogueManagerNode
from dialogue_manager.manager_node import _planner_completion_context
from dialogue_manager.manager_node import _planner_dialogue_text
from planner_common import PlannerDialogueAct
import pytest
import rclpy


@pytest.fixture(scope='module')
def rclpy_context():
    """Initialize rclpy for the test module."""
    rclpy.init()
    yield
    rclpy.shutdown()


def test_node_creation(rclpy_context):
    """Node can be created with expected name."""
    node = DialogueManagerNode()
    assert node.get_name() == 'dialogue_manager'
    node.destroy_node()


def test_parameters_declared(rclpy_context):
    """Planner and chatbot seam parameters stay declared."""
    node = DialogueManagerNode()
    assert node.has_parameter('chatbot')
    assert node.has_parameter('enable_default_chat')
    assert node.has_parameter('planner_dialogue_act_topic')
    node.destroy_node()


def test_planner_completion_context_exports_structured_facts():
    """Completion context carries structured facts for chatbot_llm wording."""
    act = PlannerDialogueAct.from_payload(
        {
            'act': 'notify_completion',
            'goal_id': 'goal_1',
            'text_hint': 'I am looking down now.',
            'context': {
                'goal_text': 'move your head up and down',
                'result_summary': 'head motion completed',
                'requested_intents': ['head_nod'],
            },
        }
    )

    payload = _planner_completion_context(act)

    assert payload['goal_id'] == 'goal_1'
    assert payload['goal_text'] == 'move your head up and down'
    assert payload['result_summary'] == 'head motion completed'
    assert payload['text_hint'] == 'I am looking down now.'
    assert payload['requested_intents'] == ['head_nod']


def test_planner_dialogue_text_prefers_structured_scan_summary_text() -> None:
    """summary_text from result_payload has priority over generic summary."""
    act = PlannerDialogueAct.from_payload(
        {
            'act': 'notify_completion',
            'goal_id': 'goal_scan_1',
            'context': {
                'result_summary': 'fallback summary text',
                'result_payload': {
                    'skill': 'scan',
                    'summary_text': 'I found one person (id: anonymous_person_1).',
                },
            },
        }
    )
    assert _planner_dialogue_text(act) == 'I found one person (id: anonymous_person_1).'

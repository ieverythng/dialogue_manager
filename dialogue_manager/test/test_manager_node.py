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
from dialogue_manager.manager_node import _planner_dialogue_context
from dialogue_manager.manager_node import _planner_dialogue_signature
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
    assert node.has_parameter('planner_dialogue_wording_mode')
    assert node.has_parameter('planner_completion_wording_mode')
    assert node.has_parameter('planner_dialogue_dedupe_window_sec')
    assert node.has_parameter('use_llm_completion_wording')
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


def test_planner_dialogue_text_turns_ask_for_help_statement_into_question() -> None:
    act = PlannerDialogueAct.from_payload(
        {
            'act': 'ask_for_help',
            'goal_id': 'goal_help_1',
            'reason': 'I could not find closest_object.',
            'text_hint': 'I could not find closest_object.',
            'await_user_response': True,
            'slots_needed': ['scan'],
        }
    )
    assert (
        _planner_dialogue_text(act)
        == 'I could not find closest_object. How should I proceed with scan?'
    )


def test_planner_dialogue_text_turns_ask_clarification_statement_into_question() -> None:
    act = PlannerDialogueAct.from_payload(
        {
            'act': 'ask_clarification',
            'goal_id': 'goal_clarify_1',
            'reason': 'I could not find cup uduka.',
            'text_hint': 'I could not find cup uduka.',
            'await_user_response': True,
        }
    )
    assert (
        _planner_dialogue_text(act)
        == 'I could not find cup uduka. Could you clarify what you want me to do next?'
    )


def test_planner_dialogue_context_exports_structured_payload() -> None:
    act = PlannerDialogueAct.from_payload(
        {
            'act': 'ask_clarification',
            'goal_id': 'goal_1',
            'goal_token': 'goal_1:__default__:1',
            'plan_id': 'plan_1',
            'plan_version': 2,
            'reason': 'target is ambiguous',
            'text_hint': 'Could you specify which cup?',
            'await_user_response': True,
            'slots_needed': ['target'],
            'context': {'goal_text': 'bring the cup'},
        }
    )

    payload = _planner_dialogue_context(act)

    assert payload['act'] == 'ask_clarification'
    assert payload['goal_id'] == 'goal_1'
    assert payload['plan_id'] == 'plan_1'
    assert payload['plan_version'] == 2
    assert payload['await_user_response'] is True
    assert payload['slots_needed'] == ['target']
    assert payload['text_hint'] == 'Could you specify which cup?'


def test_planner_dialogue_signature_stable_for_same_act() -> None:
    act = PlannerDialogueAct.from_payload(
        {
            'act': 'ask_clarification',
            'goal_id': 'goal_1',
            'goal_token': 'goal_1:tok',
            'plan_id': 'plan_1',
            'plan_version': 1,
            'text_hint': 'Which cup?',
            'reason': 'ambiguous target',
            'await_user_response': True,
        }
    )
    same_act = PlannerDialogueAct.from_payload(
        {
            'act': 'ask_clarification',
            'goal_id': 'goal_1',
            'goal_token': 'goal_1:tok',
            'plan_id': 'plan_1',
            'plan_version': 1,
            'text_hint': 'Which cup?',
            'reason': 'ambiguous target',
            'await_user_response': True,
        }
    )

    assert _planner_dialogue_signature(act) == _planner_dialogue_signature(same_act)


def test_planner_dialogue_signature_changes_for_plan_version() -> None:
    act_v1 = PlannerDialogueAct.from_payload(
        {'act': 'notify_completion', 'goal_id': 'goal_1', 'plan_id': 'plan_1', 'plan_version': 1}
    )
    act_v2 = PlannerDialogueAct.from_payload(
        {'act': 'notify_completion', 'goal_id': 'goal_1', 'plan_id': 'plan_1', 'plan_version': 2}
    )

    assert _planner_dialogue_signature(act_v1) != _planner_dialogue_signature(act_v2)

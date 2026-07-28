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
from dialogue_manager.manager_node import _planner_chatbot_context
from dialogue_manager.manager_node import _planner_dialogue_act_signature
from dialogue_manager.manager_node import _planner_safe_fallback_text
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
                'plan_outcome_summary': {
                    'completed_targets': ['apple_1', 'pear_1', 'phone_1'],
                    'failed_targets': [],
                    'pending_targets': [],
                    'all_required_steps_succeeded': True,
                },
            },
        }
    )

    payload = _planner_completion_context(act)

    assert payload['goal_id'] == 'goal_1'
    assert payload['goal_text'] == 'move your head up and down'
    assert payload['result_summary'] == 'head motion completed'
    assert payload['text_hint'] == 'I am looking down now.'
    assert payload['requested_intents'] == ['head_nod']
    assert payload['plan_outcome_summary'] == {
        'completed_targets': ['apple_1', 'pear_1', 'phone_1'],
        'failed_targets': [],
        'pending_targets': [],
        'all_required_steps_succeeded': True,
    }


def test_planner_clarification_context_routes_reason_through_chatbot():
    act = PlannerDialogueAct.from_payload(
        {
            'act': 'ask_clarification',
            'goal_id': 'goal_1',
            'plan_id': 'plan_1',
            'plan_version': 2,
            'reason': 'goal_text does not specify a physical task',
            'text_hint': 'goal_text does not specify a physical task',
            'await_user_response': True,
        }
    )

    payload = _planner_chatbot_context(act)

    assert payload['planner_dialogue']['act'] == 'ask_clarification'
    assert payload['planner_dialogue']['reason'] == (
        'goal_text does not specify a physical task'
    )
    assert payload['planner_dialogue']['await_user_response'] is True


def test_planner_act_signature_suppresses_replay_but_allows_distinct_progress() -> None:
    first = PlannerDialogueAct.from_payload(
        {
            'act': 'progress_update',
            'goal_id': 'goal_1',
            'plan_id': 'plan_1',
            'plan_version': 1,
            'text_hint': 'I am scanning the room.',
        }
    )
    replay = PlannerDialogueAct.from_payload(
        {
            'act': 'progress_update',
            'goal_id': 'goal_1',
            'plan_id': 'plan_1',
            'plan_version': 1,
            'text_hint': 'I am scanning the room.',
        }
    )
    next_progress = PlannerDialogueAct.from_payload(
        {
            'act': 'progress_update',
            'goal_id': 'goal_1',
            'plan_id': 'plan_1',
            'plan_version': 1,
            'text_hint': 'I am checking the table.',
        }
    )

    assert _planner_dialogue_act_signature(first) == _planner_dialogue_act_signature(replay)
    assert _planner_dialogue_act_signature(first) != _planner_dialogue_act_signature(next_progress)


def test_planner_act_signature_allows_distinct_clarification_slots() -> None:
    target = PlannerDialogueAct.from_payload(
        {
            'act': 'ask_clarification',
            'goal_id': 'goal_1',
            'plan_id': 'plan_1',
            'plan_version': 1,
            'slots_needed': ['target'],
        }
    )
    location = PlannerDialogueAct.from_payload(
        {
            'act': 'ask_clarification',
            'goal_id': 'goal_1',
            'plan_id': 'plan_1',
            'plan_version': 1,
            'slots_needed': ['location'],
        }
    )

    assert _planner_dialogue_act_signature(target) != _planner_dialogue_act_signature(location)


def test_planner_safe_fallback_never_exposes_raw_reason() -> None:
    act = PlannerDialogueAct.from_payload(
        {
            'act': 'explain_failure',
            'reason': 'internal planner output invalid: stack trace',
            'text_hint': 'internal planner output invalid: stack trace',
        }
    )

    assert _planner_safe_fallback_text(act) == 'I could not complete that task.'

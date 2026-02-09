# Copyright (c) 2026 IIIA-CSIC. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Integration tests with mock ROS2 nodes for dialogue_manager."""

import time

from communication_skills.action import Say
from hri_actions_msgs.msg import ClosedCaption, Intent
from hri_msgs.msg import IdsList, LiveSpeech
import pytest
import rclpy
from rclpy.action import ActionClient, ActionServer, GoalResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from tts_msgs.action import TTS


# =============================================================================
# Mock Nodes
# =============================================================================


class MockASRNode(Node):
    """Mock ASR node that publishes speech input."""

    def __init__(self):
        """Initialize the mock ASR node."""
        super().__init__('mock_asr')
        self._voices_pub = self.create_publisher(
            IdsList, '/humans/voices/tracked', 10
        )
        self._speech_pubs = {}

    def add_voice(self, voice_id: str) -> None:
        """Add a voice and publish updated tracked list."""
        if voice_id not in self._speech_pubs:
            topic = f'/humans/voices/{voice_id}/speech'
            self._speech_pubs[voice_id] = self.create_publisher(
                LiveSpeech, topic, 10
            )

        msg = IdsList()
        msg.ids = list(self._speech_pubs.keys())
        self._voices_pub.publish(msg)

    def publish_speech(self, voice_id: str, text: str) -> None:
        """Publish speech from a voice."""
        if voice_id not in self._speech_pubs:
            self.add_voice(voice_id)

        msg = LiveSpeech()
        msg.final = text
        msg.incremental = ''
        msg.locale = 'en'
        msg.confidence = 0.95
        self._speech_pubs[voice_id].publish(msg)


class MockTTSNode(Node):
    """Mock TTS engine that provides TTS action server."""

    def __init__(self):
        """Initialize the mock TTS."""
        super().__init__('mock_tts')
        self._spoken_texts = []

        self._tts_server = ActionServer(
            self,
            TTS,
            'tts_engine/tts',
            execute_callback=self._execute_tts,
            goal_callback=self._goal_callback
        )

    def _goal_callback(self, goal_request):
        """Accept all TTS goals."""
        return GoalResponse.ACCEPT

    def _execute_tts(self, goal_handle):
        """Execute TTS - simulate speaking."""
        text = goal_handle.request.input
        self._spoken_texts.append(text)

        goal_handle.succeed()
        return TTS.Result()

    def get_spoken_texts(self) -> list:
        """Return list of spoken texts."""
        return self._spoken_texts


class IntentCollector(Node):
    """Node to collect published intents."""

    def __init__(self):
        """Initialize intent collector."""
        super().__init__('intent_collector')
        self._intents = []
        self._captions = []

        self.create_subscription(
            Intent, '/intents', self._on_intent, 10
        )
        self.create_subscription(
            ClosedCaption, '/dialogue_manager/closed_captions',
            self._on_caption, 10
        )

    def _on_intent(self, msg: Intent) -> None:
        """Store received intents."""
        self._intents.append(msg)

    def _on_caption(self, msg: ClosedCaption) -> None:
        """Store received captions."""
        self._captions.append(msg)

    def get_intents(self) -> list:
        """Return list of received intents."""
        return self._intents

    def get_captions(self) -> list:
        """Return list of received captions."""
        return self._captions


class SayActionClient(Node):
    """Simple Say action client node."""

    def __init__(self):
        """Initialize Say action client."""
        super().__init__('say_client')
        self._client = ActionClient(self, Say, '/skill/say')
        self._result = None
        self._accepted = False

    def send_say(self, text: str, priority: int = 128):
        """Send Say goal."""
        if not self._client.wait_for_server(timeout_sec=2.0):
            return False

        goal = Say.Goal()
        goal.input = text
        goal.meta.priority = priority

        future = self._client.send_goal_async(goal)
        future.add_done_callback(self._goal_response_callback)
        return True

    def _goal_response_callback(self, future):
        """Handle goal response."""
        goal_handle = future.result()
        if goal_handle.accepted:
            self._accepted = True


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope='module')
def rclpy_context():
    """Initialize rclpy for the test module."""
    rclpy.init()
    yield
    rclpy.shutdown()


# =============================================================================
# Tests
# =============================================================================


class TestMockTTSNode:
    """Tests for MockTTSNode in isolation."""

    def test_mock_tts_accepts_goals(self, rclpy_context):
        """MockTTSNode accepts TTS goals."""
        mock_tts = MockTTSNode()
        client_node = Node('test_client')
        client = ActionClient(client_node, TTS, 'tts_engine/tts')

        executor = MultiThreadedExecutor()
        executor.add_node(mock_tts)
        executor.add_node(client_node)

        # Wait for server
        assert client.wait_for_server(timeout_sec=2.0)

        # Send goal
        goal = TTS.Goal()
        goal.input = 'Test speech'
        future = client.send_goal_async(goal)

        # Spin until done
        start = time.time()
        while not future.done() and time.time() - start < 5.0:
            executor.spin_once(timeout_sec=0.1)

        assert future.done()
        goal_handle = future.result()
        assert goal_handle.accepted

        # Get result
        result_future = goal_handle.get_result_async()
        while not result_future.done() and time.time() - start < 5.0:
            executor.spin_once(timeout_sec=0.1)

        assert result_future.done()
        assert 'Test speech' in mock_tts.get_spoken_texts()

        executor.shutdown()
        client_node.destroy_node()
        mock_tts.destroy_node()


class TestMockASRNode:
    """Tests for MockASRNode in isolation."""

    def test_mock_asr_publishes_voices(self, rclpy_context):
        """MockASRNode publishes voice list."""
        mock_asr = MockASRNode()
        collector = Node('collector')

        received_voices = []

        def on_voices(msg):
            received_voices.extend(msg.ids)

        collector.create_subscription(
            IdsList, '/humans/voices/tracked', on_voices, 10
        )

        executor = MultiThreadedExecutor()
        executor.add_node(mock_asr)
        executor.add_node(collector)

        # Add a voice
        mock_asr.add_voice('speaker1')

        # Spin briefly
        for _ in range(20):
            executor.spin_once(timeout_sec=0.05)

        assert 'speaker1' in received_voices

        executor.shutdown()
        collector.destroy_node()
        mock_asr.destroy_node()

    def test_mock_asr_publishes_speech(self, rclpy_context):
        """MockASRNode publishes speech messages."""
        mock_asr = MockASRNode()
        collector = Node('collector')

        received_speech = []

        def on_speech(msg):
            received_speech.append(msg.final)

        mock_asr.add_voice('speaker1')
        collector.create_subscription(
            LiveSpeech, '/humans/voices/speaker1/speech', on_speech, 10
        )

        executor = MultiThreadedExecutor()
        executor.add_node(mock_asr)
        executor.add_node(collector)

        # Allow discovery
        for _ in range(10):
            executor.spin_once(timeout_sec=0.05)

        # Publish speech
        mock_asr.publish_speech('speaker1', 'Hello world')

        # Spin to receive
        for _ in range(20):
            executor.spin_once(timeout_sec=0.05)

        assert 'Hello world' in received_speech

        executor.shutdown()
        collector.destroy_node()
        mock_asr.destroy_node()


class TestIntentCollector:
    """Tests for IntentCollector node."""

    def test_collector_receives_intents(self, rclpy_context):
        """IntentCollector receives published intents."""
        collector = IntentCollector()
        publisher_node = Node('publisher')
        intent_pub = publisher_node.create_publisher(Intent, '/intents', 10)

        executor = MultiThreadedExecutor()
        executor.add_node(collector)
        executor.add_node(publisher_node)

        # Allow discovery
        for _ in range(10):
            executor.spin_once(timeout_sec=0.05)

        # Publish intent
        intent = Intent()
        intent.intent = Intent.RAW_USER_INPUT
        intent.data = '{"test": true}'
        intent_pub.publish(intent)

        # Spin to receive
        for _ in range(20):
            executor.spin_once(timeout_sec=0.05)

        assert len(collector.get_intents()) > 0
        assert collector.get_intents()[0].intent == Intent.RAW_USER_INPUT

        executor.shutdown()
        publisher_node.destroy_node()
        collector.destroy_node()


# =============================================================================
# Full Integration Tests with DialogueManagerNode
# =============================================================================


class TestDialogueManagerSayAction:
    """Tests for Say action through DialogueManagerNode."""

    def test_say_action_triggers_tts(self, rclpy_context):
        """Say action sends text to TTS engine."""
        from dialogue_manager.manager_node import DialogueManagerNode
        from lifecycle_msgs.msg import State

        # Create nodes
        dm_node = DialogueManagerNode()
        mock_tts = MockTTSNode()
        client_node = Node('say_test_client')
        say_client = ActionClient(client_node, Say, '/skill/say')

        # Configure and activate dialogue manager
        state = State()
        dm_node.on_configure(state)
        dm_node.on_activate(state)

        executor = MultiThreadedExecutor()
        executor.add_node(dm_node)
        executor.add_node(mock_tts)
        executor.add_node(client_node)

        # Wait for server and send goal
        assert say_client.wait_for_server(timeout_sec=2.0)

        goal = Say.Goal()
        goal.input = 'Integration test speech'
        goal.meta.priority = 128
        future = say_client.send_goal_async(goal)

        # Spin until goal accepted
        start = time.time()
        while not future.done() and time.time() - start < 5.0:
            executor.spin_once(timeout_sec=0.1)

        assert future.done()
        goal_handle = future.result()
        assert goal_handle.accepted

        # Wait for TTS to receive
        result_future = goal_handle.get_result_async()
        start = time.time()
        while not result_future.done() and time.time() - start < 5.0:
            executor.spin_once(timeout_sec=0.1)

        assert 'Integration test speech' in mock_tts.get_spoken_texts()

        executor.shutdown()
        dm_node.destroy_node()
        mock_tts.destroy_node()
        client_node.destroy_node()


class TestDialogueManagerSpeechFlow:
    """Tests for speech input flow through DialogueManagerNode."""

    def test_speech_publishes_caption(self, rclpy_context):
        """Speech input publishes closed caption."""
        from dialogue_manager.manager_node import DialogueManagerNode
        from lifecycle_msgs.msg import State

        # Create nodes
        dm_node = DialogueManagerNode()
        mock_asr = MockASRNode()
        collector = IntentCollector()

        # Configure and activate dialogue manager
        state = State()
        dm_node.on_configure(state)
        dm_node.on_activate(state)

        executor = MultiThreadedExecutor()
        executor.add_node(dm_node)
        executor.add_node(mock_asr)
        executor.add_node(collector)

        # Add voice and let DM subscribe
        mock_asr.add_voice('test_speaker')
        for _ in range(30):
            executor.spin_once(timeout_sec=0.05)

        # Publish speech
        mock_asr.publish_speech('test_speaker', 'Hello dialogue manager')

        # Spin while processing
        for _ in range(50):
            executor.spin_once(timeout_sec=0.05)

        # Verify caption published
        captions = collector.get_captions()
        caption_texts = [c.text for c in captions]
        assert 'Hello dialogue manager' in caption_texts

        executor.shutdown()
        dm_node.destroy_node()
        mock_asr.destroy_node()
        collector.destroy_node()

    def test_speech_without_dialogue_publishes_intent(self, rclpy_context):
        """Speech without active dialogue publishes RAW_USER_INPUT intent."""
        from dialogue_manager.manager_node import DialogueManagerNode
        from lifecycle_msgs.msg import State

        # Create nodes
        dm_node = DialogueManagerNode()
        mock_asr = MockASRNode()
        collector = IntentCollector()

        # Configure and activate dialogue manager
        state = State()
        dm_node.on_configure(state)
        dm_node.on_activate(state)

        executor = MultiThreadedExecutor()
        executor.add_node(dm_node)
        executor.add_node(mock_asr)
        executor.add_node(collector)

        # Add voice and let DM subscribe
        mock_asr.add_voice('intent_speaker')
        for _ in range(30):
            executor.spin_once(timeout_sec=0.05)

        # Publish speech (no active dialogue)
        mock_asr.publish_speech('intent_speaker', 'Raw intent test')

        # Spin while processing
        for _ in range(50):
            executor.spin_once(timeout_sec=0.05)

        # Verify RAW_USER_INPUT intent published
        intents = collector.get_intents()
        assert len(intents) > 0
        assert any(i.intent == Intent.RAW_USER_INPUT for i in intents)

        executor.shutdown()
        dm_node.destroy_node()
        mock_asr.destroy_node()
        collector.destroy_node()


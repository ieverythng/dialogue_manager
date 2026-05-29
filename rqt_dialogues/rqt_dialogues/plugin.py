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
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""rqt plugin entry point for rqt_dialogues."""

from rqt_gui_py.plugin import Plugin

from .dialogue_view import DialogueView


class DialoguesPlugin(Plugin):
    """Live view of the Dialogue Manager's internal state."""

    def __init__(self, context):
        """Construct the plugin and attach its widget."""
        super().__init__(context)
        self.setObjectName('Dialogues')

        self._widget = DialogueView(context.node)
        if context.serial_number() > 1:
            self._widget.setWindowTitle(
                f'{self._widget.windowTitle()} ({context.serial_number()})'
            )
        context.add_widget(self._widget)

    def shutdown_plugin(self):
        """Stop subscriptions and detach the widget."""
        self._widget.shutdown()

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

"""Track ROS4HRI groups (`/humans/interactions/groups`).

Each `hri_msgs/Group` message describes a single group's current members.
The topic is latched (TRANSIENT_LOCAL) and group dispersal is signalled
by a message with an empty `members` list.
"""

from collections.abc import Callable

from hri_msgs.msg import Group
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.subscription import Subscription


GROUPS_TOPIC = '/humans/interactions/groups'


class GroupHandler:
    """Tracks current group membership and notifies on changes."""

    def __init__(
        self,
        node: Node,
        on_group_changed: Callable[[str, frozenset[str]], None] | None = None,
        on_group_dispersed: Callable[[str], None] | None = None,
    ):
        """Initialize the group handler.

        Parameters
        ----------
        node
            Lifecycle node hosting the subscription.
        on_group_changed
            Callable invoked with `(group_id, members)` whenever a group's
            membership is established or updated (members is non-empty).
        on_group_dispersed
            Callable invoked with `(group_id,)` when a group disperses
            (members becomes empty).

        """
        self._node = node
        self._on_group_changed = on_group_changed or (lambda _g, _m: None)
        self._on_group_dispersed = on_group_dispersed or (lambda _g: None)
        # group_id -> frozenset of person_ids
        self._groups: dict[str, frozenset[str]] = {}
        self._sub: Subscription | None = None

    # ------------------------------------------------------------ lifecycle

    def subscribe(self) -> None:
        """Subscribe to the groups topic (latched / TRANSIENT_LOCAL)."""
        qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._sub = self._node.create_subscription(
            Group, GROUPS_TOPIC, self._on_group, qos,
        )
        self._node.get_logger().info(f'[GROUPS] Subscribed to {GROUPS_TOPIC}')

    def unsubscribe(self) -> None:
        """Stop receiving group updates and clear the membership table."""
        if self._sub is not None:
            self._node.destroy_subscription(self._sub)
            self._sub = None
        self._groups.clear()

    # ------------------------------------------------------------ callbacks

    def _on_group(self, msg: Group) -> None:
        group_id = msg.group_id
        members = frozenset(m for m in msg.members if m)

        if not members:
            # Group dispersed. Drop from tracking and notify.
            if group_id in self._groups:
                del self._groups[group_id]
                self._node.get_logger().info(
                    f'[GROUPS] Group {group_id} dispersed'
                )
                try:
                    self._on_group_dispersed(group_id)
                except Exception as exc:
                    self._node.get_logger().warn(
                        f'[GROUPS] on_group_dispersed callback failed: {exc}'
                    )
            return

        previous = self._groups.get(group_id)
        self._groups[group_id] = members
        if previous != members:
            self._node.get_logger().info(
                f'[GROUPS] Group {group_id} now has '
                f'{len(members)} member(s): {sorted(members)}'
            )
            try:
                self._on_group_changed(group_id, members)
            except Exception as exc:
                self._node.get_logger().warn(
                    f'[GROUPS] on_group_changed callback failed: {exc}'
                )

    # ------------------------------------------------------------ queries

    def members_of(self, group_id: str) -> list[str]:
        """Return the sorted member list of `group_id` (empty if unknown)."""
        return sorted(self._groups.get(group_id, frozenset()))

    def groups_containing(self, person_id: str) -> list[str]:
        """Return all groups currently containing `person_id`."""
        return sorted(
            g for g, m in self._groups.items() if person_id in m
        )

    def co_members_of(self, person_id: str) -> set[str]:
        """Return all persons sharing at least one group with `person_id`.

        The person themselves is excluded.
        """
        co: set[str] = set()
        for members in self._groups.values():
            if person_id in members:
                co |= members
        co.discard(person_id)
        return co

    @property
    def known_groups(self) -> list[str]:
        """Return the IDs of all currently-tracked groups."""
        return sorted(self._groups.keys())

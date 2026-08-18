from dataclasses import dataclass, field
from typing import Any

from openhands.sdk.conversation.base import ConversationStateProtocol
from openhands.sdk.event import EventID
from openhands.sdk.event.base import Event
from openhands.sdk.event.llm_convertible.action import ActionEvent
from platoon.envs.base import Observation
from platoon.episode.trajectory import TrajectoryStep


@dataclass
class OpenHandsObservation(Observation):
    conversation_state: ConversationStateProtocol | None = None
    last_step_action_id: EventID | None = None
    last_step_observation_id: EventID | None = None


@dataclass
class OpenHandsAction:
    action_events: list[Event] | None = None
    misc: dict[str, Any] = field(default_factory=dict)


@dataclass
class OpenHandsTrajectoryStep(TrajectoryStep):
    action_events: list[ActionEvent] | None = None
    observation_events: list[Event] | None = None
    reward: float = 0.0

"""Test that sequence_extension prompts enable efficient training.

The key property is that for sequence_extension mode, each step's observation should be
a prefix of the next step's observation, enabling tinker to merge consecutive steps
into a single Datum.
"""

import pytest

from platoon.agents.codeact.prompt_builder import CodeActPromptBuilder
from platoon.envs.base import Task
from platoon.envs.codeact import CodeActObservation, CodeActStep


class SimplePromptBuilder(CodeActPromptBuilder):
    """Simple prompt builder with hardcoded system prompt (no Jinja templates)."""

    def build_system_prompt(self, obs: CodeActObservation, **context) -> str:
        return "You are a helpful assistant."


def create_test_observation(history_length: int) -> CodeActObservation:
    """Create a test observation with the given history length."""
    task = Task(id="test_task", goal="Solve the test problem")
    history = []
    for i in range(history_length):
        step = CodeActStep(
            thought=f"Thinking about step {i}",
            code=f"action_{i}()",
            output=f"Result of step {i}",
        )
        history.append(step)

    return CodeActObservation(
        task=task,
        action_space="You can call action_X() to perform action X.",
        history=history,
    )


def messages_to_text(messages: list) -> str:
    """Convert messages to a text representation for prefix checking."""
    parts = []
    for msg in messages:
        parts.append(f"[{msg['role']}]\n{msg['content']}")
    return "\n\n".join(parts)


class TestSequenceExtension:
    """Test that sequence_extension prompts are prefixes of subsequent prompts."""

    def test_sequence_extension_is_prefix(self):
        """Each step's prompt should be a prefix of the next step's prompt."""
        builder = SimplePromptBuilder(prompt_mode="sequence_extension")

        # Create observations with increasing history
        obs_0 = create_test_observation(0)
        obs_1 = create_test_observation(1)
        obs_2 = create_test_observation(2)

        # Build prompts
        msgs_0 = builder.build_messages(obs_0)
        msgs_1 = builder.build_messages(obs_1)
        msgs_2 = builder.build_messages(obs_2)

        # Convert to text for comparison
        text_0 = messages_to_text(msgs_0)
        text_1 = messages_to_text(msgs_1)
        text_2 = messages_to_text(msgs_2)

        # Check prefix property
        assert text_1.startswith(text_0), "Step 1's prompt should start with step 0's prompt"
        assert text_2.startswith(text_1), "Step 2's prompt should start with step 1's prompt"

    def test_no_sequence_extension_not_prefix(self):
        """no_sequence_extension prompts should NOT be prefixes (they rebuild history)."""
        builder = SimplePromptBuilder(prompt_mode="no_sequence_extension")

        obs_0 = create_test_observation(0)
        obs_1 = create_test_observation(1)

        msgs_0 = builder.build_messages(obs_0)
        msgs_1 = builder.build_messages(obs_1)

        text_0 = messages_to_text(msgs_0)
        text_1 = messages_to_text(msgs_1)

        # no_sequence_extension mode rebuilds the entire history in the user message
        # So they should NOT be prefixes (the "Action History" section changes)
        assert not text_1.startswith(text_0), "no_sequence_extension prompts should not be prefixes"

    def test_sequence_extension_grows_by_appending(self):
        """sequence_extension prompts should grow by appending, not rebuilding."""
        builder = SimplePromptBuilder(prompt_mode="sequence_extension")

        obs_1 = create_test_observation(1)
        obs_2 = create_test_observation(2)

        msgs_1 = builder.build_messages(obs_1)
        msgs_2 = builder.build_messages(obs_2)

        # Step 2 should have more messages than step 1
        assert len(msgs_2) > len(msgs_1), "More history should mean more messages"

        # The first messages should be identical
        for i in range(len(msgs_1)):
            assert msgs_1[i] == msgs_2[i], f"Message {i} should be identical"

    def test_sequence_extension_message_structure(self):
        """Verify the sequence_extension message structure."""
        builder = SimplePromptBuilder(prompt_mode="sequence_extension")

        obs_2 = create_test_observation(2)
        msgs = builder.build_messages(obs_2)

        # Expected structure:
        # [0] system
        # [1] user (initial task)
        # [2] assistant (action 0)
        # [3] user (output 0)
        # [4] assistant (action 1)
        # [5] user (output 1)

        assert len(msgs) == 6, f"Expected 6 messages, got {len(msgs)}"
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"
        assert msgs[3]["role"] == "user"
        assert msgs[4]["role"] == "assistant"
        assert msgs[5]["role"] == "user"

    def test_no_sequence_extension_message_structure(self):
        """Verify the no_sequence_extension message structure."""
        builder = SimplePromptBuilder(prompt_mode="no_sequence_extension")

        obs_2 = create_test_observation(2)
        msgs = builder.build_messages(obs_2)

        # no_sequence_extension always has exactly 2 messages
        assert len(msgs) == 2, f"Expected 2 messages, got {len(msgs)}"
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        # The user message contains all history
        assert "action_0" in msgs[1]["content"]
        assert "action_1" in msgs[1]["content"]


class TestPromptModeDefault:
    """Test that sequence_extension is the default prompt mode."""

    def test_default_is_sequence_extension(self):
        """CodeActPromptBuilder should default to sequence_extension."""
        builder = CodeActPromptBuilder()
        assert builder.prompt_mode == "sequence_extension"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

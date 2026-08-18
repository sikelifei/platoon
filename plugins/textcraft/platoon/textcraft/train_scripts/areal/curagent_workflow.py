"""AReaL workflow support for model calls made by the CurAgent harness."""

from __future__ import annotations

from typing import Any

from platoon.train.areal.workflows.step_wise import StepWiseArealWorkflow

from platoon.textcraft.curagent_areal import curagent_trace_to_trajectory_collection


class CurAgentArealWorkflow(StepWiseArealWorkflow):
    """Resolve CurAgent OpenAI calls from the AReaL session completion cache."""

    def _process_trajectory_result(
        self,
        trajectory_data: dict[str, Any] | None,
        session: Any,
        task_id: str,
        rollout_number: int,
    ) -> dict[str, Any] | None:
        if trajectory_data is not None and trajectory_data.get("adapter") == "curagent":
            completions = self.proxy_server.session_cache[session.session_id].completions
            if not completions:
                print(
                    f"[CurAgentArealWorkflow] No completions for task {task_id}, "
                    f"rollout {rollout_number}"
                )
                return None
            root_trace = (
                trajectory_data.get("curagent_trace", {})
                .get("agent_result", {})
                .get("trace")
            )
            trajectory_data = curagent_trace_to_trajectory_collection(
                root_trace,
                completions,
                task_id=task_id,
                reward=float(trajectory_data.get("reward", 0.0)),
            )
        return super()._process_trajectory_result(
            trajectory_data,
            session,
            task_id,
            rollout_number,
        )


__all__ = ["CurAgentArealWorkflow"]

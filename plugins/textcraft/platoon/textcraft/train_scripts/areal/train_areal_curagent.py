"""Train the CurAgent TextCraft-Synth harness with AReaL.

Usage:
    python -m areal.launcher.local \
      platoon/textcraft/train_scripts/areal/train_areal_curagent.py \
      --config platoon/textcraft/configs/areal/textcraft_synth_curagent_areal.yaml
"""

from __future__ import annotations

import logging
import sys
from copy import deepcopy
from dataclasses import dataclass, field

from areal.api.cli_args import load_expr_config
from datasets import Dataset
from platoon.train.areal import PlatoonArealRLTrainer, PlatoonArealRLTrainerConfig
from platoon.train.areal.config_defs import WorkflowConfig

from platoon.textcraft.curagent_areal import CurAgentRolloutConfig, run_curagent_synth_rollout
from platoon.textcraft.synth_tasks import get_synth_task
from platoon.textcraft.train_scripts.areal.curagent_workflow import CurAgentArealWorkflow
from platoon.textcraft.train_scripts.areal.train_areal_synth import get_filtered_task_ids

logger = logging.getLogger("platoon.textcraft.train_areal_curagent")


@dataclass
class CurAgentWorkflowConfig(WorkflowConfig):
    rollout_config: CurAgentRolloutConfig = field(default_factory=CurAgentRolloutConfig)


@dataclass
class CurAgentTextCraftArealConfig(PlatoonArealRLTrainerConfig):
    workflow_config: CurAgentWorkflowConfig = field(default_factory=CurAgentWorkflowConfig)
    train_difficulties: list[str] | None = None
    eval_difficulties: list[str] | None = None
    num_eval_tasks: int = 100


def reward_processor(trajectory: dict) -> tuple[float, dict[str, float]]:
    reward = float(trajectory.get("reward", 0.0))
    return reward, {"reward/success": reward}


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, CurAgentTextCraftArealConfig)
    config: CurAgentTextCraftArealConfig

    if config.workflow_config.use_subprocesses:
        raise ValueError("CurAgent AReaL rollouts currently require use_subprocesses=false")

    train_ids = get_filtered_task_ids(
        "train",
        config.train_difficulties,
        num_samples_train=2522,
    )
    eval_ids = get_filtered_task_ids(
        "val",
        config.eval_difficulties,
        num_samples_val=632,
    )[: config.num_eval_tasks]
    train_dataset = Dataset.from_list([{"task_id": task_id} for task_id in train_ids])
    val_dataset = Dataset.from_list([{"task_id": task_id} for task_id in eval_ids])
    logger.info("CurAgent train tasks: %s; eval tasks: %s", len(train_dataset), len(val_dataset))

    with PlatoonArealRLTrainer(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    ) as trainer:
        workflow = CurAgentArealWorkflow(
            run_curagent_synth_rollout,
            get_synth_task,
            config.workflow_config,
            trainer.proxy_server,
            "train_rollout",
            trainer.actor.device,
            filter_errors=True,
            reward_processor=reward_processor,
            merge_prefixes=True,
        )

        eval_config = deepcopy(config.workflow_config)
        eval_config.group_size = 1
        eval_workflow = CurAgentArealWorkflow(
            run_curagent_synth_rollout,
            get_synth_task,
            eval_config,
            trainer.eval_proxy_server,
            "eval_rollout",
            trainer.actor.device,
            reward_processor=reward_processor,
            merge_prefixes=True,
        )
        trainer.train(workflow=workflow, eval_workflow=eval_workflow)


if __name__ == "__main__":
    main(sys.argv[1:])

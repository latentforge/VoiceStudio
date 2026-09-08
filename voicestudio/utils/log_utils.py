# coding=utf-8
# Copyright 2026 LatentForge and the HuggingFace Inc. team. All rights reserved.
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
"""Where a run's readings are streamed."""

import logging
import os
from typing import Any


logger = logging.getLogger(__name__)


class BaseLogger:
    """A destination that writes nothing, which is what a run with logging turned off is given.

    Every destination here takes the step alongside the values rather than counting calls, so a
    reading taken outside the training loop lands on the same axis as the loop's own.
    """

    def log(self, metrics: dict[str, Any], step: int) -> None:
        """Writes one set of readings.

        Args:
            metrics (`dict[str, Any]`):
                The readings, keyed by name.
            step (`int`):
                The optimizer step they were taken at.
        """

    def close(self) -> None:
        """Releases whatever the destination holds open."""


class WandBLogger(BaseLogger):
    """Writes readings to a Weights and Biases run.

    Every field of the run's identity is read from the environment, so a resumed segment and a
    benchmark taken after it can be pointed at the run they belong to without the caller passing
    anything: `WANDB_TEAM`, `WANDB_PROJECT`, `WANDB_RUN_NAME`, `WANDB_RUN_ID`, `WANDB_RUN_NOTES`,
    `WANDB_RUN_TAGS`, `WANDB_RUN_GROUP`, `WANDB_RUN_JOB_TYPE`, `WANDB_RESUME_FROM` and
    `WANDB_FORK_FROM`.

    Args:
        log_dir (`str`):
            Directory the run's files are written under. Created if it does not exist.
        config (`dict[str, Any]`, *optional*):
            What the run was configured with, recorded alongside its readings.
        tag (`str`, *optional*):
            Prefix every key is filed under. A stage already prefixes its own readings, so this is
            for separating whole runs rather than the phases of one.
    """

    def __init__(self, log_dir: str, config: dict[str, Any] | None = None, tag: str | None = None):
        import wandb

        self.wandb = wandb
        self.tag = tag

        os.makedirs(log_dir, exist_ok=True)
        self.wandb.init(
            entity=os.getenv("WANDB_TEAM"),
            project=os.getenv("WANDB_PROJECT", "voicestudio"),
            name=os.getenv("WANDB_RUN_NAME"),
            id=os.getenv("WANDB_RUN_ID"),
            notes=os.getenv("WANDB_RUN_NOTES"),
            tags=os.getenv("WANDB_RUN_TAGS"),
            group=os.getenv("WANDB_RUN_GROUP"),
            job_type=os.getenv("WANDB_RUN_JOB_TYPE"),
            resume_from=os.getenv("WANDB_RESUME_FROM"),
            fork_from=os.getenv("WANDB_FORK_FROM"),
            dir=log_dir,
            config=config,
        )
        logger.info("WandB logging enabled, writing under %s", log_dir)

    def log(self, metrics: dict[str, Any], step: int) -> None:
        """Writes one set of readings under this destination's tag.

        Args:
            metrics (`dict[str, Any]`):
                The readings, keyed by name.
            step (`int`):
                The optimizer step they were taken at.
        """
        self.wandb.log(
            {(key if self.tag is None else f"{self.tag}/{key}"): value for key, value in metrics.items()},
            step=step,
        )

    def close(self) -> None:
        """Finishes the run, if one is still open."""
        if self.wandb.run is not None:
            self.wandb.finish()


__all__ = ["BaseLogger", "WandBLogger"]

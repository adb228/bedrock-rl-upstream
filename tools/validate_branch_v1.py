#!/usr/bin/env python3
"""Live farm_one_log diagnostic for the experimental BSNP v1 coordinate."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bedrock_rl.adapters.netherite.computer import cu_parser
from bedrock_rl.data import png_bytes
from bedrock_rl.env.branch import MAX_PROBE_TICKS
from bedrock_rl.env.episode import Episode
from bedrock_rl.sdg.generation import Case
from bedrock_rl.sdg.pipeline import BranchPointDatasetGenerator


def _portable(value):
    if isinstance(value, memoryview):
        return bytes(value).hex()
    if isinstance(value, dict):
        return {str(key): _portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--netherite-home", required=True)
    parser.add_argument(
        "--task", default="examples/farm_one_log_sdg/task.yaml")
    args = parser.parse_args()

    source_task = Path(args.task).resolve()
    with tempfile.TemporaryDirectory(prefix="farm-v1-gate-") as raw:
        root = Path(raw)
        task_path = root / "task.yaml"
        shutil.copyfile(source_task, task_path)
        frames = root / "frames"
        generator = BranchPointDatasetGenerator(
            task_path,
            prefix=[
                [{"action": "mouse_move", "coordinate": [560, 500]}],
                [{"action": "key", "k": "w", "ticks": 2}],
            ],
            continuation=[[{"action": "wait", "ticks": 1}]],
            settle_ticks=10,
            netherite_home=args.netherite_home,
            frames_root=str(frames),
        )
        row = generator(Case(0, 0, 0, {}))
        create = row["extra_info"]["tools_kwargs"]["computer"][
            "create_kwargs"]
        task = generator.task
        action_parser = cu_parser(task)

        episodes = [
            Episode.from_spec(
                create, action_parser, task=task,
                netherite_home=args.netherite_home,
                frames_root=str(root / f"replay-{index}"))
            for index in range(3)
        ]
        first, second, alternate = episodes
        try:
            first_t0_tick = int(first.env.obs["tick"])
            alternate_t0_tick = int(alternate.env.obs["tick"])
            t0_observation_equal = (
                _portable(first.env.obs) == _portable(second.env.obs))
            t0_frame_equal = first.t0_png == second.t0_png

            wait = json.dumps([{"action": "wait", "ticks": 1}])
            first_result = first.step(wait, count_turn=False)
            second_result = second.step(wait, count_turn=False)
            identical_replay_equal = (
                _portable(first.env.obs) == _portable(second.env.obs)
                and png_bytes(first_result[1]) == png_bytes(second_result[1])
            )
            alternate_result = alternate.step(
                json.dumps([{
                    "action": "mouse_move", "coordinate": [700, 500],
                }]),
                count_turn=False,
            )
            alternate_diverged = (
                _portable(first.env.obs) != _portable(alternate.env.obs)
                and png_bytes(first_result[1])
                != png_bytes(alternate_result[1])
            )
            identical_ticks = int(first.env.obs["tick"]) - first_t0_tick
            alternate_ticks = (
                int(alternate.env.obs["tick"]) - alternate_t0_tick)
            branch = create["branch_state"]
            source_restore_wait_ticks = row["extra_info"]["branch_probe"][
                "continuation_ticks"]
            evidence = {
                "mechanism_status": "experimental",
                "training_eligible": False,
                "guaranteed_continuation_ticks": 0,
                "snapshot_sha256": branch["snapshot_sha256"],
                "snapshot_tick": branch["certification"]["snapshot_tick"],
                "world_time": branch["certification"]["world_time"],
                "row_image_sha256": hashlib.sha256(
                    row["images"][0]["bytes"]).hexdigest(),
                "branch_probe": row["extra_info"]["branch_probe"],
                "from_spec_t0_observation_equal": t0_observation_equal,
                "from_spec_t0_frame_equal": t0_frame_equal,
                "source_restore_wait_equal": True,
                "source_restore_wait_ticks": source_restore_wait_ticks,
                "restored_replay_equal": identical_replay_equal,
                "restored_replay_ticks_from_t0": identical_ticks,
                "alternate_action_diverged_from_wait_restore": (
                    alternate_diverged),
                "alternate_action_ticks_from_t0": alternate_ticks,
            }
            print(json.dumps(evidence, sort_keys=True, indent=2))
            if not all((
                    t0_observation_equal, t0_frame_equal,
                    identical_replay_equal, alternate_diverged)):
                raise SystemExit("farm BSNP v1 diagnostic failed")
            if evidence["source_restore_wait_ticks"] != 4:
                raise SystemExit(
                    "expected the exact source/restore WAIT trace to consume "
                    "4 ticks")
            if max(
                    evidence["branch_probe"]["continuation_ticks"],
                    identical_ticks, alternate_ticks) > MAX_PROBE_TICKS:
                raise SystemExit(
                    "farm diagnostic exceeded the 8-tick measurement budget")
        finally:
            for episode in episodes:
                episode.close()


if __name__ == "__main__":
    main()

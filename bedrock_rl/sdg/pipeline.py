"""Netherite components for the generic data-generation runner."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Mapping
from uuid import uuid4

import yaml

from bedrock_rl.adapters.netherite.chat import IMAGE_TAG
from bedrock_rl.adapters.netherite.computer import (
    NEXT_COMPUTER_MESSAGE, TOOL_NAME, cu_parser, cu_tool_call, cu_user_prompt,
)
from bedrock_rl.sdg.policy import PolicyVeto
from bedrock_rl.core.components import ComponentSpec, resolve_callable
from bedrock_rl.sdg.generation import (Case, CaseRejected, assign_path,
                                           case_value)
from bedrock_rl.core.messages import data_image_part, text_part
from bedrock_rl.core.reward import RewardResult
from bedrock_rl.core.state import StateEvent, json_value
from bedrock_rl.core.tools import ToolCall, ToolResult
from bedrock_rl.core.trajectory import TrajectoryRecorder, TrajectoryView
from bedrock_rl.data import png_bytes
from bedrock_rl.sdg.policy.oracle import expert_cu
from bedrock_rl.env.engine import DEFAULT_VIEW_DISTANCE, EngineTimeout
from bedrock_rl.env.branch import (
    MAX_PROBE_TICKS,
    branch_output_dir,
    inspect_quiescent_snapshot,
    make_branch_state,
    snapshot_sha256,
)
from bedrock_rl.env.episode import (Episode, episode_spec, retain_frames_dir,
                                      tool_response_text)
from bedrock_rl.env.snapshots import SnapshotRejected
from bedrock_rl.sdg.policy.spatial import PlanRejected
from bedrock_rl.sdg.policy.world import (
    SnapshotBoundsError, SnapshotWorldMap,
)
from bedrock_rl.env.task import Task, relative_path

DATA_SOURCE = "bedrock"
INTERFACE = "cu"


class ExpertPolicy:
    """Expose the registered task oracle as a closed-loop SDG policy.

    ``ScriptedEpisodeGenerator`` owns execution and recording. This policy
    only asks the task's registered oracle for the next public computer action
    after each new observation, so any supported task can produce multi-turn
    demonstrations without adding an example-local wrapper.
    """

    def reset(self, task, episode, case):
        del task, episode
        self.rng = random.Random(case.decision_seed)

    def actions(self, task, episode, case, turn):
        del case, turn
        return expert_cu(task, episode.env, self.rng)


def observable_action_summary(observation, actions, turn):
    """Describe emitted public-interface actions without exposing a plan."""
    del observation, turn
    kinds = {str(action.get("action", "")) for action in actions}
    phrases = []
    if "mouse_move" in kinds:
        phrases.append("adjust my view")
    if "key" in kinds:
        phrases.append("move or select an item")
    if kinds & {"left_click", "right_click", "left_click_hold"}:
        phrases.append("interact with what I am aiming at")
    if "wait" in kinds:
        phrases.append("wait for the next observation")
    if not phrases:
        phrases.append("wait for the next observation")
    return "From the current observation, I will " + ", then ".join(
        phrases) + "."


def dataset_path(task, mode, hint_level="none"):
    """Default parquet path shared by generation and training configs."""
    if mode not in ("rl", "sft"):
        raise ValueError("dataset mode must be 'rl' or 'sft'")
    if mode == "rl" and hint_level != "none":
        raise ValueError(
            "RL prompt datasets must be hint-free; use SDPO or the "
            "self-distillation workflow for removable teacher guidance")
    if mode == "rl":
        filename = "rl_prompts.parquet"
    else:
        filename = "sft.parquet"
    return os.path.join("data", f"{task.name}_{INTERFACE}", filename)


def load_episode_specs(path):
    """Read deterministic engine coordinates from an RL prompt parquet."""
    import pandas as pd

    frame = pd.read_parquet(path, columns=["extra_info"])
    return [episode_spec(info["task_yaml"], info["seed"],
                         info["init_yaw"], info["init_pitch"],
                         info.get("branch_state"))
            for info in frame["extra_info"]]


def _path(value, base_dir):
    path = Path(value)
    return path if path.is_absolute() else Path(base_dir) / path




class EpisodeDatasetGenerator:
    """One engine-backed RL prompt or one-turn expert SFT row per case."""

    def __init__(self, task, mode="rl", hint_level="none",
                 netherite_home=None, frames_root=None, *, base_dir=".",
                 metadata=None, data_source=DATA_SOURCE):
        self.task_path = _path(task, base_dir).resolve()
        self.task = Task.load(str(self.task_path))
        self.mode = str(mode)
        if self.mode not in ("rl", "sft"):
            raise ValueError("EpisodeDatasetGenerator mode is rl or sft")
        self.hint_level = str(hint_level)
        if self.mode == "rl" and self.hint_level != "none":
            raise ValueError(
                "RL prompt rows cannot contain static hints; use SDPO or "
                "self-distillation to keep guidance teacher-only")
        self.netherite_home = netherite_home
        self.frames_root = (frames_root or os.path.join(
            "/tmp", f"bedrock-sdg-{self.task.name}"))
        self.metadata = dict(metadata or {})
        self.data_source = str(data_source)
        if not self.data_source:
            raise ValueError("data_source cannot be empty")

    def __call__(self, case):
        from bedrock_rl.sdg.guidance import hint_text, with_hint
        rng = random.Random(case.decision_seed)
        episode = Episode.draw(
            self.task, case.seed, rng, parser=cu_parser(self.task),
            netherite_home=self.netherite_home,
            frames_root=self.frames_root)
        try:
            prompt = cu_user_prompt(self.task, episode.env)
            spec = dict(episode.spec,
                        task_yaml=relative_path(episode.spec["task_yaml"]),
                        interface=INTERFACE)
            if self.mode == "rl":
                spec["need_tools_kwargs"] = True
                spec["tools_kwargs"] = {TOOL_NAME: {"create_kwargs": dict(
                    episode.spec, task_yaml=spec["task_yaml"],
                    interface=INTERFACE)}}
                return {
                    "data_source": self.data_source,
                    "prompt": with_hint(
                        [{"role": "user", "content": prompt}],
                        hint_text(self.task, self.hint_level)),
                    "ability": "game",
                    "reward_model": {"style": "rule", "ground_truth": ""},
                    "extra_info": spec,
                    "agent_name": "bedrock",
                    "images": [{"bytes": episode.t0_png, "path": None}],
                }
            actions = expert_cu(self.task, episode.env, rng)
            if not actions:
                return None
            episode.step(json.dumps(actions))
            if not episode.success:
                return None
            return {
                "question": prompt, "answer": cu_tool_call(actions),
                "reward": float(episode.final_reward()),
                "seed": int(case.seed),
                "images": [{"bytes": episode.t0_png, "path": None}],
            }
        finally:
            episode.close()


class BranchPointDatasetGenerator:
    """Write one RL row from an empirically probed BSNP v1 branch point.

    This component accepts only movement/look/wait controls, settles to a
    motionless on-ground state, and checks the state BSNP v1 represents.
    ``continuation`` is mandatory: source and restore execute those exact
    public turns and must produce identical complete observations, CPU frames,
    and final BSNP v1 bytes within a bounded diagnostic run. BSNP v1 omits
    hidden live state, so only that listed trace passed and the guaranteed
    continuation window is zero.
    """

    thread_safe = False
    _ACTION_KEYS = {
        "mouse_move": frozenset(("action", "coordinate")),
        "wait": frozenset(("action", "ticks")),
        "key": frozenset(("action", "k", "key", "ticks")),
    }
    _MOVEMENT_KEYS = frozenset((
        "w", "a", "s", "d", "space", "sneak", "sprint",
    ))

    def __init__(self, task, prefix, continuation, settle_ticks=10,
                 netherite_home=None, frames_root=None, *, base_dir=".",
                 metadata=None, data_source=DATA_SOURCE):
        self.task_path = _path(task, base_dir).resolve()
        self.task = Task.load(str(self.task_path))
        world = self.task.initial_state.world
        if world.get("mobs") is not False:
            raise ValueError(
                "BSNP v1 branch tasks must explicitly declare "
                "initial_state.world.mobs: false")
        if "time" not in world:
            raise ValueError(
                "BSNP v1 branch tasks require an explicit "
                "initial_state.world.time")
        if self.task.raw.get("raster", "cpu") != "cpu":
            raise ValueError("BSNP v1 branch tasks require raster 'cpu'")
        view_distance = self.task.raw.get("view_distance")
        if view_distance is None:
            view_distance = DEFAULT_VIEW_DISTANCE
        if isinstance(view_distance, bool) or not isinstance(
                view_distance, int):
            raise TypeError(
                "BSNP v1 branch task view_distance must be an integer")
        if not 1 <= view_distance <= 8:
            raise ValueError(
                "BSNP v1 branch task view_distance must be in 1..8")
        self.prefix = self._controls(prefix, "prefix")
        self.continuation = self._controls(continuation, "continuation")
        if isinstance(settle_ticks, bool) or not isinstance(settle_ticks, int):
            raise TypeError("branch settle_ticks must be an integer")
        if not 2 <= settle_ticks <= 200:
            raise ValueError("branch settle_ticks must be in 2..200")
        self.settle_ticks = int(settle_ticks)
        if len(self.prefix) > self.task.max_turns:
            raise ValueError(
                f"branch prefix has {len(self.prefix)} turns but task "
                f"max_turns is {self.task.max_turns}")
        self.branches_dir = branch_output_dir(self.task.path)
        self.netherite_home = netherite_home
        self.frames_root = (frames_root or os.path.join(
            "/tmp", f"bedrock-branch-sdg-{self.task.name}"))
        self.metadata = dict(metadata or {})
        self.data_source = str(data_source)
        if not self.data_source:
            raise ValueError("data_source cannot be empty")

    @classmethod
    def _controls(cls, value, what):
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError(
                f"BranchPointDatasetGenerator {what} must be a non-empty "
                "list of turns")
        normalized = []
        for turn_index, turn in enumerate(value):
            if not isinstance(turn, (list, tuple)) or not turn:
                raise TypeError(
                    f"branch {what} turn {turn_index} must be a non-empty "
                    "list of action mappings")
            if any(not isinstance(action, Mapping) for action in turn):
                raise TypeError(
                    f"branch {what} turn {turn_index} contains a non-mapping "
                    "action")
            copied = []
            for action in turn:
                action = copy.deepcopy(dict(action))
                name = action.get("action")
                if name not in cls._ACTION_KEYS:
                    raise ValueError(
                        f"branch {what} permits only movement/look/wait "
                        f"actions, not {name!r}")
                unknown = sorted(set(action) - cls._ACTION_KEYS[name])
                if unknown:
                    raise ValueError(
                        f"branch {what} {name} action has unknown keys: "
                        + ", ".join(unknown))
                if name == "key":
                    if "k" in action and "key" in action:
                        raise ValueError(
                            f"branch {what} key action cannot supply both "
                            "'k' and 'key'")
                    keys = action.get("k", action.get("key"))
                    keys = keys if isinstance(keys, list) else [keys]
                    lowered = [str(key).lower() for key in keys]
                    if not lowered or any(
                            key not in cls._MOVEMENT_KEYS for key in lowered):
                        raise ValueError(
                            f"branch {what} key actions permit only "
                            "w/a/s/d/space/sneak/sprint")
                copied.append(action)
            normalized.append(copied)
        json.dumps(normalized, allow_nan=False, sort_keys=True)
        return normalized

    @staticmethod
    def _portable_observation(episode):
        def portable(value):
            if isinstance(value, memoryview):
                return {"__bytes__": bytes(value).hex()}
            if isinstance(value, Mapping):
                return {str(key): portable(item)
                        for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [portable(item) for item in value]
            return value
        return portable(episode.env.obs)

    @classmethod
    def _environment_observation_hash(cls, env):
        body = json.dumps(
            cls._portable_observation(
                type("_EpisodeView", (), {"env": env})()),
            sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(body.encode()).hexdigest()

    @classmethod
    def _observation_hash(cls, episode):
        body = json.dumps(
            cls._portable_observation(episode), sort_keys=True,
            separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(body.encode()).hexdigest()

    @staticmethod
    def _checked_step(episode, actions, label):
        penalty = episode.penalty
        result = episode.step(
            json.dumps(actions, separators=(",", ":")), count_turn=False)
        if episode.penalty != penalty:
            raise CaseRejected(
                f"{label} was malformed or penalized instead of executing",
                code="branch_malformed_control")
        return result

    def _write_snapshot(self, episode, case):
        self.branches_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.branches_dir / (
            f".branch.{os.getpid()}.{int(case.index)}.{uuid4().hex}.bsnp")
        try:
            episode.env.snapshot(
                temporary, radius=self.task.snapshot_radius)
            digest = snapshot_sha256(temporary)
            destination = self.branches_dir / f"{digest}.bsnp"
            if destination.exists():
                if snapshot_sha256(destination) != digest:
                    raise RuntimeError(
                        f"content-addressed branch snapshot {destination} "
                        "does not match its filename")
                temporary.unlink()
            else:
                os.replace(temporary, destination)
            inspect_quiescent_snapshot(destination)
            return destination
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def __call__(self, case):
        rng = random.Random(case.decision_seed)
        parser = cu_parser(self.task)
        source_tick_hashes = []
        probe_tick_hashes = []
        source = Episode.draw(
            self.task, case.seed, rng, parser=parser,
            netherite_home=self.netherite_home,
            frames_root=self.frames_root, raster="cpu",
            capture_frames=True,
            render_set={"anim_textures": 0, "no_hand": 1},
            random_ticks=False,
            tick_observer=lambda env: source_tick_hashes.append(
                self._environment_observation_hash(env)))
        branch_spec = None
        probe = None
        branch_home = None
        probe_evidence = None
        try:
            for turn, actions in enumerate(self.prefix):
                self._checked_step(
                    source, actions, f"branch prefix turn {turn + 1}")
                if source.done:
                    raise CaseRejected(
                        f"branch prefix reached a terminal state at turn "
                        f"{turn + 1}; a branch point must remain mid-episode",
                        code="branch_prefix_terminal")
            self._checked_step(
                source,
                [{"action": "wait", "ticks": self.settle_ticks}],
                "branch settling turn")
            if source.done:
                raise CaseRejected(
                    "branch settling reached a terminal task state",
                    code="branch_settle_terminal")

            # BSNP v1 omits both world-clock counters.  The exact original
            # engine identity makes elapsed header ticks a stable clock
            # alignment rule: --set establishes the task time at process
            # boot, then every observation tick advances world_time once.
            world_time = (
                int(self.task.initial_state.world["time"])
                + int(source.env.obs["tick"])
                - int(source.env.engine_tick_origin)
            )
            snapshot = self._write_snapshot(source, case)
            state = make_branch_state(
                snapshot, self.task.path,
                netherite_home=source.netherite_home,
                raster=source.raster,
                view_distance=source.view_distance,
                world_time=world_time,
                settle_ticks=self.settle_ticks)
            branch_home = source.netherite_home
            branch_spec = episode_spec(
                self.task.path, case.seed,
                source.env.obs["yaw"], source.env.obs["pitch"], state)
            # snapshot() advances the source one idle tick.  Its next frame
            # tick is therefore aligned with branch reset_pose + t0.
            source_t0 = png_bytes(source.frame())
            probe = Episode.from_spec(
                branch_spec, parser, task=self.task,
                netherite_home=source.netherite_home,
                frames_root=self.frames_root, raster="cpu",
                view_distance=source.view_distance,
                tick_observer=lambda env: probe_tick_hashes.append(
                    self._environment_observation_hash(env)))
            source_hash = self._observation_hash(source)
            probe_hash = self._observation_hash(probe)
            if source_hash != probe_hash or source_t0 != probe.t0_png:
                source_obs = self._portable_observation(source)
                probe_obs = self._portable_observation(probe)
                differing = sorted(
                    key for key in set(source_obs) | set(probe_obs)
                    if source_obs.get(key) != probe_obs.get(key))
                pixel_detail = ""
                if source_t0 and probe.t0_png:
                    from PIL import Image, ImageChops
                    source_image = Image.open(io.BytesIO(source_t0)).convert(
                        "RGB")
                    probe_image = Image.open(
                        io.BytesIO(probe.t0_png)).convert("RGB")
                    difference = ImageChops.difference(
                        source_image, probe_image)
                    pixel_detail = (
                        f", pixel_bbox={difference.getbbox()}")
                raise CaseRejected(
                    "source continuation and reopened branch disagree at t0 "
                    f"(obs {source_hash[:12]} != {probe_hash[:12]}, frames "
                    f"{hashlib.sha256(source_t0 or b'').hexdigest()[:12]} != "
                    f"{hashlib.sha256(probe.t0_png or b'').hexdigest()[:12]}, "
                    f"fields {differing}{pixel_detail})",
                    code="branch_source_mismatch")
            probe_start_tick = int(source.env.obs["tick"])
            source_tick_hashes.clear()
            probe_tick_hashes.clear()
            for turn, actions in enumerate(self.continuation):
                source_at = len(source_tick_hashes)
                probe_at = len(probe_tick_hashes)
                source_result = self._checked_step(
                    source, actions,
                    f"source continuation turn {turn + 1}")
                probe_result = self._checked_step(
                    probe, actions,
                    f"reopened continuation turn {turn + 1}")
                if (source_tick_hashes[source_at:] !=
                        probe_tick_hashes[probe_at:]
                        or self._observation_hash(source)
                        != self._observation_hash(probe)
                        or png_bytes(source_result[1])
                        != png_bytes(probe_result[1])):
                    raise CaseRejected(
                        "source continuation and reopened branch diverged "
                        f"after probe turn {turn + 1}",
                        code="branch_source_mismatch")
            probe_end_tick = int(source.env.obs["tick"])
            continuation_ticks = probe_end_tick - probe_start_tick
            if continuation_ticks > MAX_PROBE_TICKS:
                raise CaseRejected(
                    "BSNP v1 continuation probe exceeded the diagnostic "
                    f"{MAX_PROBE_TICKS}-tick measurement budget "
                    f"({continuation_ticks} ticks)",
                    code="branch_probe_too_long")
            # Compare every byte BSNP v1 can represent.  This deliberately
            # does not claim to compare omitted plant RNG/growth, scheduled
            # updates, live counters, or clocks.
            with tempfile.TemporaryDirectory(
                    prefix="bedrock-branch-v1-gate-") as directory:
                source_end = Path(directory) / "source.bsnp"
                probe_end = Path(directory) / "reopened.bsnp"
                source.env.snapshot(
                    source_end, radius=self.task.snapshot_radius)
                probe.env.snapshot(
                    probe_end, radius=self.task.snapshot_radius)
                source_digest = snapshot_sha256(source_end)
                probe_digest = snapshot_sha256(probe_end)
                if source_end.read_bytes() != probe_end.read_bytes():
                    raise CaseRejected(
                        "source continuation and reopened branch BSNP v1 "
                        "represented state diverged after the probe",
                        code="branch_source_mismatch")
                probe_evidence = {
                    "continuation_turns": len(self.continuation),
                    "continuation_sha256": hashlib.sha256(json.dumps(
                        self.continuation, sort_keys=True, allow_nan=False,
                        separators=(",", ":")).encode()).hexdigest(),
                    "source_tick_start": probe_start_tick,
                    "source_tick_end": probe_end_tick,
                    "continuation_ticks": continuation_ticks,
                    "per_tick_observations": len(source_tick_hashes),
                    "max_probe_ticks": MAX_PROBE_TICKS,
                    "snapshot_format": "BSNP/1",
                    "represented_state_sha256": source_digest,
                    "restored_state_sha256": probe_digest,
                    "scope": (
                        "exact diagnostic trace only; zero guaranteed "
                        "continuation ticks"),
                }
        finally:
            if probe is not None:
                probe.close()
            source.close()

        episode = Episode.from_spec(
            branch_spec, parser, task=self.task,
            netherite_home=branch_home,
            frames_root=self.frames_root, raster="cpu",
            view_distance=state["runtime"]["view_distance"])
        try:
            prompt = cu_user_prompt(self.task, episode.env)
            spec = dict(
                episode.spec,
                task_yaml=relative_path(episode.spec["task_yaml"]),
                interface=INTERFACE,
            )
            spec["need_tools_kwargs"] = True
            spec["branch_probe"] = probe_evidence
            spec["branch_contract"] = {
                "status": "experimental",
                "training_eligible": False,
                "guaranteed_continuation_ticks": 0,
            }
            spec["tools_kwargs"] = {TOOL_NAME: {"create_kwargs": dict(
                episode.spec, task_yaml=spec["task_yaml"],
                interface=INTERFACE)}}
            return {
                "data_source": self.data_source,
                "prompt": [{"role": "user", "content": prompt}],
                "ability": "game",
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": spec,
                "agent_name": "bedrock",
                "images": [{"bytes": episode.t0_png, "path": None}],
            }
        finally:
            episode.close()


def _message_content(frame, text, *, include_image=True):
    content = []
    if include_image:
        encoded = png_bytes(frame)
        if encoded is not None:
            content.append(data_image_part(encoded))
    content.append(text_part(text))
    return content


def _episode_state(episode, semantic_camera=False):
    """Dynamic, bounded state for one scripted episode transition.

    Episode reconstruction coordinates are static and live once in trajectory
    provenance. Repeating them here made long trajectories carry hundreds of
    identical task paths and view specifications.
    """
    from bedrock_rl.adapters.netherite.probes import (
        DEFAULT_OBSERVATION_FIELDS, capture_observation,
    )
    fields = list(DEFAULT_OBSERVATION_FIELDS)
    if semantic_camera:
        fields.extend(("cam", "depth", "edge"))
    return {
        "episode": {
            "done": bool(episode.done), "success": bool(episode.success),
            "failed": bool(episode.failed), "turns": int(episode.turns),
            "actions": int(episode.nlines),
            "reward": float(episode.final_reward()),
        },
        "observation": capture_observation(episode, fields),
    }


def _trajectory_episode_spec(episode):
    """One portable reconstruction spec for trajectory provenance."""

    spec = copy.deepcopy(episode.spec)
    spec["task_yaml"] = relative_path(spec["task_yaml"])
    return spec


def _record_new_events(episode, recorder, event_index):
    events = episode.env.journal[event_index:]
    for event in events:
        value = event.as_dict()
        name = value.pop("event")
        tick = value.pop("tick")
        recorder.trajectory.state.extend_events(
            [StateEvent(name, value, int(tick))])
    return event_index + len(events)


def _qualification_requirements(value, what):
    """Normalize named minimum-count requirements without a block registry."""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{what} must be a mapping")
    normalized = {}
    for raw_name, declaration in value.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError(f"{what} names cannot be empty")
        if not isinstance(declaration, Mapping):
            raise TypeError(f"{what}.{name} must be a mapping")
        unknown = sorted(set(declaration) - {"blocks", "minimum"})
        if unknown:
            raise ValueError(
                f"unknown {what}.{name} keys: " + ", ".join(unknown))
        blocks = declaration.get("blocks")
        if isinstance(blocks, (str, int)):
            blocks = (blocks,)
        elif blocks is not None:
            blocks = tuple(blocks)
        if not blocks:
            raise ValueError(f"{what}.{name}.blocks cannot be empty")
        minimum = declaration.get("minimum")
        if (isinstance(minimum, bool) or not isinstance(minimum, int)
                or minimum < 1):
            raise ValueError(
                f"{what}.{name}.minimum must be a positive integer")
        normalized[name] = {"blocks": blocks, "minimum": minimum}
    return normalized


class SnapshotQualifier:
    """Cheap declarative filters over one immutable Minecraft snapshot.

    Resource and source-fluid minima are counted with the snapshot map's
    vectorized packed-voxel queries. Optional import-path checks extend the
    filter without teaching the generation framework about portals, trees,
    arenas, or any other task. A failed check raises :class:`CaseRejected`,
    so exact-N generation simply advances to the next deterministic seed.

    A custom check implements ``qualify(task, world, case)`` (or is directly
    callable with those arguments). It must be read-only and should reject
    expected natural-world misses with ``CaseRejected``.
    """

    def __init__(self, resources=None, source_fluids=None, checks=None, *,
                 base_dir=".", metadata=None):
        del metadata
        self.resources = _qualification_requirements(
            resources, "qualification resources")
        self.source_fluids = _qualification_requirements(
            source_fluids, "qualification source_fluids")
        self.check_specs = tuple(
            ComponentSpec.parse(item, what="snapshot qualification check")
            for item in (checks or ()))
        self.checks = tuple(spec.create(base_dir=base_dir)
                            for spec in self.check_specs)
        self._count_cache = None

    @staticmethod
    def _missing(requirements, counts):
        return {
            name: {"available": int(counts[name]),
                   "required": int(declaration["minimum"]),
                   "blocks": list(declaration["blocks"])}
            for name, declaration in requirements.items()
            if counts[name] < declaration["minimum"]
        }

    def _counts(self, world):
        key = (
            world.snapshot_sha256,
            tuple((name, tuple(value["blocks"]), value["minimum"])
                  for name, value in self.resources.items()),
            tuple((name, tuple(value["blocks"]), value["minimum"])
                  for name, value in self.source_fluids.items()),
        )
        cached = self._count_cache
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
        resource_counts = ({name: 0 for name in self.resources}
                           if not self.resources else
                           world.count_blocks_many(
                               {name: value["blocks"]
                                for name, value in self.resources.items()},
                               limits={name: value["minimum"]
                                       for name, value
                                       in self.resources.items()}))
        fluid_counts = ({name: 0 for name in self.source_fluids}
                        if not self.source_fluids else
                        world.count_source_fluids_many(
                            {name: value["blocks"] for name, value
                             in self.source_fluids.items()},
                            limits={name: value["minimum"] for name, value
                                    in self.source_fluids.items()}))
        # Retain only one world's tiny count summary. Long seed scouts remain
        # bounded while samples_per_world variants avoid rescanning voxels.
        self._count_cache = (key, resource_counts, fluid_counts)
        return resource_counts, fluid_counts

    def qualify(self, task, snapshot, case):
        with SnapshotWorldMap(snapshot) as world:
            resource_counts, fluid_counts = self._counts(world)
            missing = self._missing(self.resources, resource_counts)
            if missing:
                raise CaseRejected(
                    "snapshot is missing required natural blocks: "
                    + ", ".join(
                        f"{name}={item['available']}/{item['required']}"
                        for name, item in missing.items()),
                    code="qualification_resource_missing",
                    details={"missing": missing,
                             "snapshot_sha256": world.snapshot_sha256})
            missing = self._missing(self.source_fluids, fluid_counts)
            if missing:
                raise CaseRejected(
                    "snapshot is missing required source fluids: "
                    + ", ".join(
                        f"{name}={item['available']}/{item['required']}"
                        for name, item in missing.items()),
                    code="qualification_source_fluid_missing",
                    details={"missing": missing,
                             "snapshot_sha256": world.snapshot_sha256})
            check_results = {}
            for index, check in enumerate(self.checks):
                function = getattr(check, "qualify", check)
                if not callable(function):
                    raise TypeError(
                        "snapshot qualification check must be callable or "
                        "implement qualify(task, world, case)")
                value = function(task, world, case)
                if value is not None:
                    check_results[str(index)] = json_value(
                        value, what="snapshot qualification result")
            return {
                "snapshot_sha256": world.snapshot_sha256,
                "resources": dict(resource_counts),
                "source_fluids": dict(fluid_counts),
                "checks": check_results,
            }


class WorldFunctionCheck:
    """Adapt a read-only world function into a qualification component.

    The world is passed as the first positional argument. ``bindings`` maps
    function keyword arguments to deterministic case/world fields. Supported
    sources are ordinary :func:`case_value` names plus ``player_cell`` and
    ``world_seed``. This keeps structure checks import-path modular instead
    of adding structure names to the generator.
    """

    def __init__(self, function, bindings=None, kwargs=None,
                 code="qualification_world_check", label=None, **_):
        self.function = resolve_callable(
            function, what="qualification world function")
        self.bindings = dict(bindings or {})
        self.kwargs = dict(kwargs or {})
        self.code = str(code)
        self.label = str(label or getattr(
            self.function, "__qualname__", "world check"))

    @staticmethod
    def _binding(source, world, case):
        if source == "player_cell":
            player = world.player
            return (math.floor(player.x), math.floor(player.y),
                    math.floor(player.z))
        if source == "world_seed":
            return int(world.seed)
        return case_value(case, source)

    def qualify(self, task, world, case):
        del task
        arguments = copy.deepcopy(self.kwargs)
        for target, source in self.bindings.items():
            arguments[str(target)] = self._binding(str(source), world, case)
        try:
            self.function(world, **arguments)
        except PlanRejected as exc:
            raise CaseRejected(
                f"{self.label} rejected the snapshot: {exc}",
                code=self.code,
                details={"check": self.label}) from exc
        return {"check": self.label, "accepted": True}


class ScriptedEpisodeGenerator:
    """Scout seeds by running a closed-loop script to declared success.

    The policy is an import-path component with an optional ``reset`` method
    and an ``actions(task, episode, case, turn)`` method. It emits the same
    mouse/keyboard action objects accepted by the public ``computer`` tool.
    A case is accepted only when the task's real live-state success check
    fires; failed worlds are ordinary generator rejections, allowing
    ``max_world_attempts`` and a process executor to find exactly the
    requested number of usable worlds in parallel. ``task_bindings`` resolves
    case values into arbitrary task fields before snapshot selection.
    """

    # Qualification and task resolution cache per-world state. Process and
    # serial executors give that state one owner; sharing this instance across
    # threads lets concurrent worlds overwrite each other's cache entries.
    thread_safe = False

    def __init__(self, task, policy, netherite_home=None, frames_root=None,
                 include_images=True, record_views=False,
                 semantic_camera=False, recording_every=None,
                 max_turns=None, task_bindings=None, qualifier=None,
                 headless_qualification=False, decision_attempts=1,
                 assistant_content=None, *,
                 base_dir=".", metadata=None):
        self.task_path = _path(task, base_dir).resolve()
        with self.task_path.open() as stream:
            self.task_template = yaml.safe_load(stream)
        if not isinstance(self.task_template, Mapping):
            raise TypeError("scripted task template must be a mapping")
        self.task_template = copy.deepcopy(dict(self.task_template))
        self.task_bindings = dict(task_bindings or {})
        self.decision_attempts = int(decision_attempts)
        if self.decision_attempts < 1:
            raise ValueError("scripted generator decision_attempts must be positive")
        if (self.decision_attempts > 1
                and "decision_seed" in self.task_bindings.values()):
            raise ValueError(
                "decision_attempts cannot retry a decision seed that is "
                "bound into the task document")
        self.task = Task(copy.deepcopy(self.task_template),
                         path=str(self.task_path))
        self.policy = ComponentSpec.parse(policy, what="scripted policy")
        self.policy.resolve()
        self.assistant_content = (
            None if assistant_content is None else resolve_callable(
                assistant_content, what="scripted assistant_content"))
        self.qualifier = (None if qualifier is None else
                          ComponentSpec.parse(
                              qualifier, what="scripted qualifier").create(
                                  base_dir=base_dir,
                                  metadata=dict(metadata or {})))
        self.netherite_home = netherite_home
        self.include_images = bool(include_images)
        self.headless_qualification = bool(headless_qualification)
        self.record_views = bool(record_views)
        self.semantic_camera = bool(semantic_camera)
        self.recording_every = (None if not recording_every
                                else int(recording_every))
        if self.recording_every is not None:
            if self.recording_every < 1:
                raise ValueError("recording_every must be positive")
        self.frames_root = (frames_root or os.path.join(
            "/tmp", f"bedrock-sdg-{self.task.name}"))
        self.max_turns_override = (None if max_turns is None
                                   else int(max_turns))
        self.max_turns = (self.task.max_turns if max_turns is None
                          else self.max_turns_override)
        if self.max_turns < 1:
            raise ValueError("scripted generator max_turns must be positive")
        self.metadata = dict(metadata or {})
        self._artifact_owner_pid = os.getpid()
        self._resolved_task = None
        self._qualified_world = None
        self._qualified_cases = {}
        self._qualification_snapshot = None

    def _task_document(self, case):
        template = getattr(self, "task_template", None)
        if template is None:
            return None
        document = copy.deepcopy(template)
        for target, source in getattr(self, "task_bindings", {}).items():
            assign_path(document, target, case_value(case, source))
        return document

    def _task_and_digest(self, case):
        document = self._task_document(case)
        if document is None:
            return self.task, "unavailable"
        body = json.dumps(document, sort_keys=True, allow_nan=False,
                          separators=(",", ":"))
        digest = hashlib.sha256(body.encode()).hexdigest()
        cached = getattr(self, "_resolved_task", None)
        if cached is not None and cached[0] == digest:
            return cached[1], digest
        task = Task(document, path=str(self.task_path))
        # Only the current world's task is retained. A large seed scout may
        # see thousands of distinct world bindings, so an unbounded task cache
        # would defeat the generation runner's bounded-memory contract.
        self._resolved_task = (digest, task)
        return task, digest

    def group_identity(self, case):
        """The complete task identity that must be shared by one world."""
        _, digest = self._task_and_digest(case)
        return {"world_seed": int(case.seed), "task_digest": digest}

    @staticmethod
    def _qualification_key(case, task_digest):
        return (int(case.seed), str(task_digest), int(case.decision_seed))

    def _qualify_case(self, task, task_digest, case):
        qualifier = getattr(self, "qualifier", None)
        if qualifier is None:
            return None
        world_key = (int(case.seed), str(task_digest))
        if getattr(self, "_qualified_world", None) != world_key:
            self._qualified_world = world_key
            self._qualified_cases = {}
            self._qualification_snapshot = None
        key = self._qualification_key(case, task_digest)
        if key in self._qualified_cases:
            return self._qualified_cases[key]
        started = time.perf_counter()
        try:
            snapshot = task.snapshot_for(
                case.seed, netherite_home=self.netherite_home)
            if not snapshot:
                raise CaseRejected(
                    "snapshot qualification requires initial_state",
                    code="qualification_snapshot_missing")
            function = getattr(qualifier, "qualify", qualifier)
            if not callable(function):
                raise TypeError(
                    "scripted qualifier must be callable or implement "
                    "qualify(task, snapshot, case)")
            result = function(task, snapshot, case)
        except EngineTimeout as exc:
            raise CaseRejected(
                str(exc), code="engine_timeout",
                details={"phase": "qualification"}) from exc
        except SnapshotRejected as exc:
            raise CaseRejected(
                str(exc), code="world_constraint") from exc
        except CaseRejected as exc:
            elapsed = time.perf_counter() - started
            raise CaseRejected(
                f"{exc}; qualification_seconds={elapsed:.6f}",
                code=exc.code,
                details={
                    **dict(exc.details),
                    "timing_seconds": {"qualification": round(elapsed, 6)},
                }) from exc
        elapsed = round(time.perf_counter() - started, 6)
        self._qualification_snapshot = str(snapshot)
        value = {"seconds": elapsed, "result": result}
        self._qualified_cases[key] = value
        return value

    def qualify_group(self, cases):
        """Qualify every decision variant before executing any of them."""

        if getattr(self, "qualifier", None) is None:
            return
        for case in cases:
            task, task_digest = self._task_and_digest(case)
            self._qualify_case(task, task_digest, case)

    def _run_case(self, case, *, include_images, recording_every,
                  record_views):
        started_at = time.perf_counter()
        task, task_digest = self._task_and_digest(case)
        qualification = self._qualify_case(task, task_digest, case)
        if (recording_every is not None
                and task.view.type not in (
                    "netherite-procedural", "minecraft-official")):
            raise ValueError("dense recording is available only for "
                             "exact engine-pixel task views")
        max_turns = (getattr(
                        task, "max_turns", getattr(self, "max_turns", 1))
                     if getattr(self, "max_turns_override", None) is None
                     else self.max_turns_override)
        rng = random.Random(case.decision_seed)
        draw_started_at = time.perf_counter()
        try:
            episode = Episode.draw(
                task, case.seed, rng, parser=cu_parser(task),
                netherite_home=self.netherite_home,
                frames_root=(self.frames_root if (include_images
                             or recording_every) else None),
                capture_every=recording_every,
                capture_frames=(include_images or
                                recording_every is not None))
        except SnapshotRejected as exc:
            elapsed = time.perf_counter() - draw_started_at
            raise CaseRejected(
                f"{exc}; episode_draw_seconds={elapsed:.6f}",
                code="world_constraint",
                details={"timing_seconds": {
                    "episode_draw": round(elapsed, 6),
                }}) from exc
        if qualification is not None:
            qualified_snapshot = getattr(
                self, "_qualification_snapshot", None)
            episode_snapshot = getattr(episode.env, "snapshot_in", None)
            if (qualified_snapshot is None or episode_snapshot is None
                    or os.path.realpath(qualified_snapshot)
                    != os.path.realpath(episode_snapshot)):
                episode.close()
                raise RuntimeError(
                    "episode did not restore the snapshot that passed world "
                    "qualification")
        draw_seconds = time.perf_counter() - draw_started_at
        policy = None
        try:
            if isinstance(getattr(episode, "spec", None), dict):
                episode.spec["task_digest"] = task_digest
            policy = self.policy.create()
            reset = getattr(policy, "reset", None)
            preflight_started_at = time.perf_counter()
            try:
                if reset is not None:
                    reset(task, episode, case)
            except CaseRejected as exc:
                preflight_seconds = (time.perf_counter()
                                     - preflight_started_at)
                raise CaseRejected(
                    f"{exc}; episode_draw_seconds={draw_seconds:.6f}; "
                    f"policy_preflight_seconds={preflight_seconds:.6f}",
                    code=exc.code,
                    details={
                        **dict(getattr(exc, "details", {})),
                        "timing_seconds": {
                            "episode_draw": round(draw_seconds, 6),
                            "policy_preflight": round(
                                preflight_seconds, 6),
                        },
                    }) from exc
            policy_provenance = getattr(policy, "provenance", None)
            policy_plan = (policy_provenance()
                           if policy_provenance is not None else None)
            if policy_plan is not None and not isinstance(
                    policy_plan, Mapping):
                raise TypeError(
                    "scripted policy provenance() must return a mapping")
            policy_cost_fn = getattr(policy, "cost_certificate", None)
            policy_cost = (policy_cost_fn()
                           if policy_cost_fn is not None else None)
            if policy_cost is not None and not isinstance(
                    policy_cost, Mapping):
                raise TypeError(
                    "scripted policy cost_certificate() must return a "
                    "mapping")
            grounding_fn = getattr(policy, "demonstration_grounding", None)
            grounding = (grounding_fn() if grounding_fn is not None else {
                "eligible": None,
                "target_selection": "undeclared",
                "execution": "undeclared",
            })
            if not isinstance(grounding, Mapping):
                raise TypeError(
                    "scripted policy demonstration_grounding() must return "
                    "a mapping")
            preflight_seconds = time.perf_counter() - preflight_started_at
            timing = {
                "episode_draw": round(draw_seconds, 6),
                "policy_preflight": round(preflight_seconds, 6),
            }
            if qualification is not None:
                timing["qualification"] = qualification["seconds"]
            recorder = TrajectoryRecorder(
                task={"name": task.name, "instruction": task.goal,
                      "path": relative_path(str(self.task_path))},
                provenance={"generator": type(self).__module__ + ":"
                            + type(self).__qualname__,
                            "policy": self.policy.type,
                            "task_digest": task_digest,
                            "episode_spec": _trajectory_episode_spec(episode),
                            "view": episode.view_provenance,
                            "demonstration_grounding": dict(grounding),
                            **({"policy_plan": dict(policy_plan)}
                               if policy_plan is not None else {}),
                            **({"policy_cost_certificate": dict(policy_cost)}
                               if policy_cost is not None else {})},
                metadata={"generation_case": case.to_dict(),
                          "generation_timing_seconds": timing,
                          **self.metadata})
            prompt = cu_user_prompt(task, episode.env)
            if prompt.startswith(IMAGE_TAG):
                # The image is represented by a typed part in trajectories,
                # but the separator after it is model-facing prompt content.
                # Remove exactly the marker and preserve every following byte.
                prompt = prompt[len(IMAGE_TAG):]
            recorder.append({
                "role": "user",
                "content": _message_content(
                    episode.t0_png, prompt,
                    include_image=include_images),
            })
            recorder.record_state(
                0, "reset", _episode_state(episode, self.semantic_camera))
            event_index = _record_new_events(episode, recorder, 0)
            public_observation = prompt
            progression_started_at = time.perf_counter()
            recent_actions = []
            for turn in range(max_turns):
                if record_views:
                    recorder.trajectory.views.append(TrajectoryView(
                        "script", turn, copy.deepcopy(recorder.messages),
                        [{"type": "scripted_policy",
                          "policy": self.policy.type,
                          "decision_seed": int(case.decision_seed)}]))
                fn = getattr(policy, "actions", policy)
                actions = fn(task, episode, case, turn)
                if isinstance(actions, PolicyVeto):
                    veto_timing = {
                        **timing,
                        "live_progression": round(
                            time.perf_counter() - progression_started_at, 6),
                        "total_to_verification": round(
                            time.perf_counter() - started_at, 6),
                    }
                    diagnose = getattr(policy, "diagnostics", None)
                    raise CaseRejected(
                        f"{actions.reason}; target={actions.target}, "
                        f"observed={actions.observed}",
                        code=actions.code,
                        details={
                            "target": actions.target,
                            "observed": actions.observed,
                            "timing_seconds": veto_timing,
                            "recent_actions": recent_actions,
                            **({"policy_state": diagnose(episode.env)}
                               if diagnose is not None else {}),
                        })
                if not actions:
                    break
                turn_grounding_fn = getattr(policy, "turn_grounding", None)
                turn_grounding = (
                    turn_grounding_fn() if turn_grounding_fn is not None
                    else grounding)
                if not isinstance(turn_grounding, Mapping):
                    raise TypeError(
                        "scripted policy turn_grounding() must return a "
                        "mapping")
                recent_actions.append({
                    "turn": int(turn),
                    "stage": str(getattr(policy, "stage", "unknown")),
                    "position": tuple(round(float(episode.env.obs[key]), 3)
                                      for key in ("x", "y", "z")),
                    "actions": tuple(
                        (action.get("action"), action.get("k"),
                         action.get("ticks"))
                        for action in actions),
                })
                recent_actions = recent_actions[-16:]
                call = ToolCall(f"script_{turn}", TOOL_NAME,
                                {"actions": copy.deepcopy(actions)})
                content = None
                if self.assistant_content is not None:
                    content = self.assistant_content(
                        public_observation, copy.deepcopy(actions), turn)
                    if content is not None and not isinstance(content, str):
                        raise TypeError(
                            "scripted assistant_content must return text "
                            "or None")
                recorder.append({
                    "role": "assistant", "content": content,
                    "tool_calls": [call.to_openai()],
                    "metadata": {
                        "policy_stage": str(getattr(
                            policy, "stage", "unknown")),
                        "demonstration_grounding": dict(turn_grounding),
                    },
                })
                observation, frame, step_reward, done = episode.step(
                    json.dumps(actions))
                public_observation = observation
                result = ToolResult(
                    _message_content(
                        frame, tool_response_text(
                            observation, done, NEXT_COMPUTER_MESSAGE),
                        include_image=include_images),
                    {"turns": episode.turns, "success": episode.success,
                     "step_reward": float(step_reward)},
                    done=bool(done), reward=float(step_reward),
                    attachments=({"image": frame}
                                 if include_images else {}))
                recorder.record_tool_result(turn, call, result)
                recorder.extend(result.messages(call.id, name=call.name))
                recorder.record_state(turn + 1, "after_tools",
                                      _episode_state(
                                          episode, self.semantic_camera))
                event_index = _record_new_events(
                    episode, recorder, event_index)
                if done:
                    break
            progression_seconds = (time.perf_counter()
                                   - progression_started_at)
            timing.update({
                "live_progression": round(progression_seconds, 6),
                "total_to_verification": round(
                    time.perf_counter() - started_at, 6),
            })
            recorder.trajectory.metadata[
                "generation_timing_seconds"] = dict(timing)
            if not episode.success:
                stage = getattr(policy, "stage", "unknown")
                inventory = {
                    name: int(count) for name, count in
                    episode.env.obs["inv_counts"].items() if count
                }
                diagnose = getattr(policy, "diagnostics", None)
                policy_state = (diagnose(episode.env)
                                if diagnose is not None else {})
                recent_events = [event.as_dict() for event in
                                 episode.env.journal[-5:]]
                recent_fluid = [event.as_dict() for event in
                                episode.env.journal
                                if event.name == "player_fluid_state"][-5:]
                raise CaseRejected(
                    f"live success not reached; stage={stage}, "
                    f"turns={episode.turns}, actions={episode.nlines}, "
                    f"ticks={episode.env.ticks}, done={episode.done}, "
                    f"failed={episode.failed}, "
                    f"timing_seconds={timing}, "
                    f"inventory={inventory}, policy_state={policy_state}, "
                    f"recent_actions={recent_actions}, "
                    f"recent_fluid={recent_fluid}, "
                    f"recent_events={recent_events}",
                    details={
                        "timing_seconds": timing,
                        "policy_state": policy_state,
                        "stage": stage,
                        "recent_actions": recent_actions,
                    })
            score = float(episode.final_reward())
            reward = RewardResult(
                score, {"bedrock": score},
                {"success": True, "turns": int(episode.turns)})
            if recording_every and episode._frames_dir:
                source = Path(episode._frames_dir)
                preserved = source.with_name(source.name + ".recording")
                source.replace(preserved)
                # The directory is now an artifact awaiting the writer, not
                # live scratch owned by this worker. A replacement process
                # must not reap it merely because the producing PID exited.
                retain_frames_dir(
                    preserved, owner_pid=self._artifact_owner_pid)
                episode._frames_dir = None
                recorder.trajectory.metadata["_recording_frames_dir"] = str(
                    preserved)
                recorder.trajectory.metadata["recording_every"] = int(
                    recording_every)
            return recorder.finish(reward)
        finally:
            try:
                close = getattr(policy, "close", None)
                if close is not None:
                    close()
            finally:
                episode.close()

    @staticmethod
    def _action_digest(trajectory):
        """Stable digest of the exact computer controls in a trajectory."""

        actions = []
        for message in trajectory.messages:
            if not isinstance(message, Mapping):
                continue
            for call in message.get("tool_calls") or ():
                function = call.get("function") or {}
                if function.get("name") != TOOL_NAME:
                    continue
                arguments = function.get("arguments", "")
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                actions.append(arguments.get("actions"))
        body = json.dumps(actions, sort_keys=True, allow_nan=False,
                          separators=(",", ":"))
        return hashlib.sha256(body.encode()).hexdigest(), len(actions)

    def _materialize_case(self, case):
        """Qualify one exact decision, then render only after it succeeds."""

        qualify_headless = (
            self.include_images
            and getattr(self, "headless_qualification", False))
        if not qualify_headless:
            return self._run_case(
                case, include_images=self.include_images,
                recording_every=self.recording_every,
                record_views=getattr(self, "record_views", False))

        qualified = self._run_case(
            case, include_images=False, recording_every=None,
            record_views=False)
        qualified_digest, qualified_turns = self._action_digest(qualified)
        qualified_timing = dict(qualified.metadata.get(
            "generation_timing_seconds", {}))
        rendered = self._run_case(
            case, include_images=True,
            recording_every=self.recording_every,
            record_views=getattr(self, "record_views", False))
        rendered_digest, rendered_turns = self._action_digest(rendered)
        if (rendered_digest, rendered_turns) != (
                qualified_digest, qualified_turns):
            raise CaseRejected(
                "headless qualification and rendered materialization "
                "produced different computer controls",
                code="render_materialization_diverged",
                details={
                    "headless_action_sha256": qualified_digest,
                    "headless_turns": qualified_turns,
                    "rendered_action_sha256": rendered_digest,
                    "rendered_turns": rendered_turns,
                })
        rendered.metadata["headless_qualification"] = {
            "action_sha256": qualified_digest,
            "turns": qualified_turns,
            "timing_seconds": qualified_timing,
        }
        return rendered

    @staticmethod
    def _retry_decision_case(case, attempt):
        """Derive another policy seed without changing world or variables."""

        attempt = int(attempt)
        if attempt == 0:
            return case
        body = (f"netherite-decision-retry-v1:{int(case.seed)}:"
                f"{int(case.decision_seed)}:{attempt}")
        decision_seed = int.from_bytes(
            hashlib.sha256(body.encode()).digest()[:8], "big") % (2**63 - 1)
        if decision_seed == int(case.decision_seed):
            decision_seed = (decision_seed + 1) % (2**63 - 1)
        return Case(
            int(case.index), int(case.seed), decision_seed,
            copy.deepcopy(case.values),
            world_index=case.world_index, sample_index=case.sample_index,
            world_values=copy.deepcopy(case.world_values))

    @staticmethod
    def _decision_retryable(rejection):
        """Whether another ordering can change this exact-world rejection."""

        code = str(rejection.code)
        return not (code in {
                        "world_constraint",
                        "world_plan_invalid_start",
                        "world_plan_bounds",
                        "world_plan_resource_unreachable",
                        "world_plan_search_budget",
                    }
                    or code == "engine_timeout"
                    or code.startswith("qualification_"))

    def __call__(self, case):
        """Try bounded policy seeds, retaining one exact accepted world."""

        attempts = int(getattr(self, "decision_attempts", 1))
        rejected = []
        for attempt in range(attempts):
            selected = self._retry_decision_case(case, attempt)
            try:
                trajectory = self._materialize_case(selected)
            except SnapshotBoundsError as exc:
                raise CaseRejected(
                    str(exc), code="world_plan_bounds",
                    details={
                        "attempt": attempt,
                        "decision_seed": int(selected.decision_seed),
                    }) from exc
            except EngineTimeout as exc:
                raise CaseRejected(
                    str(exc), code="engine_timeout",
                    details={
                        "attempt": attempt,
                        "decision_seed": int(selected.decision_seed),
                    }) from exc
            except CaseRejected as exc:
                rejected.append({
                    "attempt": attempt,
                    "decision_seed": int(selected.decision_seed),
                    "code": exc.code,
                })
                if (attempt + 1 >= attempts
                        or not self._decision_retryable(exc)):
                    raise CaseRejected(
                        str(exc), code=exc.code,
                        details={
                            **dict(exc.details),
                            "decision_selection": {
                                "attempts": rejected,
                                "configured_attempts": attempts,
                            },
                        }) from exc
                continue
            if attempts > 1:
                trajectory.metadata["decision_selection"] = {
                    "original_decision_seed": int(case.decision_seed),
                    "selected_decision_seed": int(selected.decision_seed),
                    "selected_attempt": attempt,
                    "rejected_attempts": rejected,
                    "configured_attempts": attempts,
                }
            return trajectory
        raise AssertionError("decision retry loop produced no outcome")




class TaskVariantGenerator:
    """Turn each case into a complete task document with no hardcoded schema.

    Like ``TrialGenerator``, ``bindings`` maps dotted document targets to
    case fields (``seed``, ``decision_seed``, or sampled value names). Static
    overrides use the same dotted targets and are applied last. Unknown task
    fields are rejected by the task loader, so generation cannot smuggle in
    a setting the environment ignores.
    """

    def __init__(self, template, bindings=None, overrides=None,
                 name="{name}_{index:04d}", *, base_dir=".", metadata=None):
        self.template_path = _path(template, base_dir).resolve()
        with self.template_path.open() as stream:
            self.template = yaml.safe_load(stream)
        self.bindings = dict(bindings or {})
        self.overrides = dict(overrides or {})
        self.name = str(name)
        self.metadata = dict(metadata or {})

    def __call__(self, case):
        task = copy.deepcopy(self.template)
        original = str(task.get("name") or "task")
        task["name"] = self.name.format(
            name=original, index=case.index, seed=case.seed,
            **case.values)
        task["seeds"] = [int(case.seed)]
        for dotted, source in self.bindings.items():
            assign_path(task, dotted, case_value(case, source))
        for dotted, value in self.overrides.items():
            assign_path(task, dotted, value)
        # Validate immediately so malformed variants fail during generation,
        # not when a later training job opens the output.
        Task(task, path=str(self.template_path))
        return {
            "schema": "bedrock.task-instance.v1",
            "case": case.to_dict(),
            "task": task,
            "provenance": {"template": str(self.template_path),
                           "metadata": self.metadata},
        }

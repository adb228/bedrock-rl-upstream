import io
import json
import struct
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from bedrock_rl.adapters.netherite.environment import NetheriteEnvironment
from bedrock_rl.data import write_rows
from bedrock_rl.env.branch import (
    BRANCH_STATE_SCHEMA,
    CERTIFICATION_PROFILE,
    canonical_branch_state,
    materialized_branch_snapshot,
    resolve_branch_snapshot,
    snapshot_sha256,
    task_sha256,
)
from bedrock_rl.env.episode import Episode, _open_env, episode_spec, spec_values
from bedrock_rl.env.snapshots import ANGLE_OFF
from bedrock_rl.env.task import Check, Task
from bedrock_rl.sdg.generation import Case
from bedrock_rl.sdg.pipeline import BranchPointDatasetGenerator


IDENTITY = {
    "actual": {
        "head": "1" * 40,
        "manifest": {
            "source": "recorded",
            "patches": {"0001-test.diff": "2" * 64},
        },
    },
    "expected": {
        "head": "1" * 40,
        "manifest": {"0001-test.diff": "2" * 64},
    },
    "binary": {"name": "magma_game", "sha256": "3" * 64},
}


def _state(reference, digest, *, task_digest="4" * 64, tick=12,
           world_time=1012, view_distance=8, identity=IDENTITY):
    return {
        "schema": BRANCH_STATE_SCHEMA,
        "snapshot_ref": reference,
        "snapshot_sha256": digest,
        "task_sha256": task_digest,
        "engine_identity": identity,
        "runtime": {
            "raster": "cpu", "view_distance": view_distance,
            "render_set": {"anim_textures": 0, "no_hand": 1},
            "random_ticks": False,
        },
        "certification": {
            "profile": CERTIFICATION_PROFILE,
            "snapshot_format": "BSNP/1",
            "action_domain": "movement-look-wait",
            "snapshot_tick": tick,
            "world_time": world_time,
            "settle_ticks": 10,
            "max_probe_ticks": 8,
        },
    }


def _png(color=(12, 34, 56)):
    output = io.BytesIO()
    Image.new("RGB", (2, 2), color).save(output, format="PNG")
    return output.getvalue()


class BranchReferenceTests(unittest.TestCase):
    def test_legacy_four_coordinates_keep_their_exact_shape(self):
        spec = episode_spec("task.yaml", 7, 12, -3)
        self.assertEqual(spec, {
            "task_yaml": "task.yaml", "seed": 7,
            "init_yaw": 12.0, "init_pitch": -3.0,
        })
        self.assertEqual(
            spec_values(spec), ("task.yaml", 7, 12.0, -3.0))

    def test_experimental_fifth_coordinate_round_trips(self):
        digest = "a" * 64
        state = _state(f"datasets/branch_snapshots/{digest}.bsnp", digest)
        spec = episode_spec("task.yaml", 7, 12, -3, state)
        self.assertEqual(spec["branch_state"], state)
        self.assertIsNot(spec["branch_state"], state)
        json.dumps(spec, allow_nan=False)

    def test_path_syntax_and_digest_basename_are_strict(self):
        digest = "a" * 64
        unsafe = (
            f"/tmp/{digest}.bsnp",
            f"../{digest}.bsnp",
            f"branches/../{digest}.bsnp",
            fr"branches\{digest}.bsnp",
            f"branches//{digest}.bsnp",
            f"branches/./{digest}.bsnp",
            "branches/state.bsnp",
        )
        for reference in unsafe:
            with self.subTest(reference=reference), self.assertRaises(
                    ValueError):
                canonical_branch_state(_state(reference, digest))

    def test_runtime_and_settle_bounds_are_closed(self):
        digest = "a" * 64
        reference = f"datasets/branch_snapshots/{digest}.bsnp"
        for distance in (0, 9):
            with self.subTest(view_distance=distance), self.assertRaises(
                    ValueError):
                canonical_branch_state(
                    _state(reference, digest, view_distance=distance))
        for settle in (1, 201):
            state = _state(reference, digest)
            state["certification"]["settle_ticks"] = settle
            with self.subTest(settle_ticks=settle), self.assertRaises(
                    ValueError):
                canonical_branch_state(state)
        state = _state(reference, digest)
        state["certification"]["max_probe_ticks"] = 9
        with self.assertRaisesRegex(ValueError, "at most 8"):
            canonical_branch_state(state)
        state = _state(reference, digest)
        state["runtime"]["random_ticks"] = True
        with self.assertRaisesRegex(ValueError, "random_ticks false"):
            canonical_branch_state(state)

    def test_later_snapshot_profiles_are_not_runtime_dependencies(self):
        digest = "a" * 64
        reference = f"datasets/branch_snapshots/{digest}.bsnp"
        state = _state(reference, digest)
        state["schema"] = "bedrock.branch-state.v2"
        state["certification"].update({
            "profile": "netherite.quiescent-cpu.v2",
            "snapshot_format": "BSNP/2+LIV2/1",
        })
        with self.assertRaisesRegex(ValueError, "unsupported branch_state"):
            canonical_branch_state(state)

    def test_consumer_rechecks_explicit_task_prerequisites(self):
        digest = "a" * 64
        reference = f"datasets/branch_snapshots/{digest}.bsnp"
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory) / "task.yaml"
            for world in ("{time: 1000}", "{mobs: false}"):
                task.write_text(
                    "name: test\ngoal: wait\n"
                    f"initial_state: {{world: {world}}}\n"
                    "success: {type: pitch_below, value: -80}\n")
                state = _state(
                    reference, digest, task_digest=task_sha256(task))
                with self.subTest(world=world), self.assertRaisesRegex(
                        RuntimeError, "explicit"):
                    resolve_branch_snapshot(
                        task, state, netherite_home="/engine",
                        render_set={"anim_textures": 0, "no_hand": 1})

    def test_header_detectable_motion_rejects_hand_built_snapshot(self):
        from tests.test_snapshot_cache import write_spawn_snapshot

        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "state.bsnp"
            write_spawn_snapshot(snapshot)
            data = bytearray(snapshot.read_bytes())
            motion_off = ANGLE_OFF + struct.calcsize("<2f")
            struct.pack_into("<d", data, motion_off, 0.25)
            snapshot.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "motion"):
                from bedrock_rl.env.branch import inspect_quiescent_snapshot
                inspect_quiescent_snapshot(snapshot)

    def test_resolver_checks_task_content_binary_identity_and_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "task.yaml"
            task.write_text(
                "name: test\ngoal: wait\ninitial_state:\n"
                "  world: {time: 1000, mobs: false}\n"
                "success: {type: pitch_below, value: -80}\n")
            snapshot = root / "datasets" / "branch_snapshots" / \
                f"{'a' * 64}.bsnp"
            snapshot.parent.mkdir(parents=True)
            snapshot.write_bytes(b"controlled snapshot")
            digest = snapshot_sha256(snapshot)
            renamed = snapshot.with_name(f"{digest}.bsnp")
            snapshot.rename(renamed)
            state = _state(
                f"datasets/branch_snapshots/{digest}.bsnp", digest,
                task_digest=task_sha256(task))
            inspected = {"snapshot_tick": 12}

            with patch("bedrock_rl.env.branch._engine_identity",
                       return_value=IDENTITY), patch(
                           "bedrock_rl.env.branch.inspect_quiescent_snapshot",
                           return_value=inspected):
                self.assertEqual(
                    resolve_branch_snapshot(
                        task, state, netherite_home="/engine",
                        render_set={"anim_textures": 0, "no_hand": 1}),
                    str(renamed))
                with materialized_branch_snapshot(
                        task, state, netherite_home="/engine", raster="cpu",
                        view_distance=8,
                        render_set={"anim_textures": 0, "no_hand": 1}
                        ) as (private, canonical):
                    self.assertNotEqual(private, str(renamed))
                    self.assertEqual(Path(private).read_bytes(),
                                     b"controlled snapshot")
                    self.assertEqual(canonical, state)
                    private_path = Path(private)
                self.assertFalse(private_path.exists())

                task.write_text(
                    "name: changed\ngoal: wait\ninitial_state:\n"
                    "  world: {time: 1000, mobs: false}\n"
                    "success: {type: pitch_below, value: -80}\n")
                with self.assertRaisesRegex(RuntimeError, "task digest"):
                    resolve_branch_snapshot(
                        task, state, netherite_home="/engine",
                        render_set={"anim_textures": 0, "no_hand": 1})
                task.write_text(
                    "name: test\ngoal: wait\ninitial_state:\n"
                    "  world: {time: 1000, mobs: false}\n"
                    "success: {type: pitch_below, value: -80}\n")
                renamed.write_bytes(b"corrupted")
                with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                    resolve_branch_snapshot(
                        task, state, netherite_home="/engine",
                        render_set={"anim_textures": 0, "no_hand": 1})

    def test_symlink_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "task.yaml"
            task.write_text(
                "name: test\ngoal: wait\ninitial_state:\n"
                "  world: {time: 1000, mobs: false}\n"
                "success: {type: pitch_below, value: -80}\n")
            target = root / "target"
            target.write_bytes(b"x")
            digest = snapshot_sha256(target)
            link = root / "datasets" / "branch_snapshots" / \
                f"{digest}.bsnp"
            link.parent.mkdir(parents=True)
            link.symlink_to(target)
            state = _state(
                f"datasets/branch_snapshots/{link.name}", digest,
                task_digest=task_sha256(task))
            with patch("bedrock_rl.env.branch._engine_identity",
                       return_value=IDENTITY), self.assertRaisesRegex(
                           ValueError, "must not be a symlink"):
                resolve_branch_snapshot(
                    task, state, netherite_home="/engine",
                    render_set={"anim_textures": 0, "no_hand": 1})

    def test_engine_identity_hashes_selected_binary_bytes(self):
        from bedrock_rl.env import branch
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "magma_game"
            binary.write_bytes(b"the actual executable")
            expected = snapshot_sha256(binary)
            manifest = {
                name: "2" * 64 for name in branch.APPROVED_PATCH_STACK
            }
            with patch("bedrock_rl.env.engine.find_binary",
                       return_value=binary), patch(
                           "bedrock_rl.env.snapshots.engine_cache_identity",
                           return_value={
                               "actual": {
                                   "head": "1" * 40,
                                   "manifest": {
                                       "source": "recorded",
                                       "patches": manifest,
                                   },
                               },
                               "expected": {
                                   "head": "1" * 40,
                                   "manifest": manifest,
                               },
                           }), patch(
                               "bedrock_rl.env.engine.patch_status",
                               return_value={
                                   "adopted": list(manifest),
                                   "problems": [],
                               }), patch(
                                   "bedrock_rl.env.branch.subprocess.run",
                                   return_value=SimpleNamespace(
                                       returncode=0, stderr="", stdout="")):
                identity = branch._engine_identity("/selected", "cpu")
        self.assertEqual(identity["binary"], {
            "name": "magma_game",
            "sha256": expected,
        })
        self.assertTrue(identity["approval"]["exact_original_0012_stack"])

    def test_open_env_uses_private_copy_and_coordinate_clock(self):
        task = SimpleNamespace(
            name="branch", path="/task.yaml", snapshot_for=Mock())
        digest = "a" * 64
        state = _state(
            f"datasets/branch_snapshots/{digest}.bsnp", digest,
            world_time=2468)

        @contextmanager
        def materialized(*_args, **_kwargs):
            yield "/private/verified.bsnp", state

        with patch("bedrock_rl.env.episode.materialized_branch_snapshot",
                   side_effect=materialized) as materialize, patch(
                       "bedrock_rl.env.episode.MagmaEnv",
                       return_value="env") as magma:
            self.assertEqual(
                _open_env(
                    task, 7, branch_state=state, netherite_home="/engine",
                    raster="cpu", view_distance=5,
                    render_set={"anim_textures": 0, "no_hand": 1}), "env")
        task.snapshot_for.assert_not_called()
        self.assertEqual(magma.call_args.kwargs["snapshot_in"],
                         "/private/verified.bsnp")
        self.assertEqual(magma.call_args.kwargs["world_time"], 2468)
        self.assertEqual(materialize.call_args.kwargs["view_distance"], 5)


class NewEpisodeSemanticsTests(unittest.TestCase):
    @staticmethod
    def _task(path):
        path.write_text("\n".join((
            "name: branch_test",
            "goal: Move.",
            "seeds: [7]",
            "view: semantic",
            "max_turns: 4",
            "initial_state:",
            "  world: {time: 1000, mobs: false}",
            "success: {type: pitch_below, value: -80}",
        )))
        return Task.load(str(path))

    def test_task_argument_must_match_branch_spec_task_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            one = Path(directory) / "one.yaml"
            two = Path(directory) / "two.yaml"
            task = self._task(one)
            self._task(two)
            digest = "a" * 64
            state = _state(
                f"datasets/branch_snapshots/{digest}.bsnp", digest)
            with self.assertRaisesRegex(ValueError, "does not match"):
                Episode.from_spec(
                    episode_spec(two, 7, 0, 0, state), object(), task=task)

    def test_branch_runtime_conflict_is_rejected_before_open(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.yaml"
            task = self._task(path)
            digest = "a" * 64
            state = _state(
                f"datasets/branch_snapshots/{digest}.bsnp", digest,
                view_distance=6)
            spec = {**episode_spec(path, 7, 0, 0, state),
                    "runtime": state["runtime"]}
            with self.assertRaisesRegex(RuntimeError, "conflicts"):
                Episode.from_spec(
                    spec, object(), task=task, view_distance=8)

    def test_t0_refreshes_delta_and_convergence_baselines(self):
        class FakeEnv:
            def __init__(self):
                self.obs = {
                    "x": 4.5, "y": 64.0, "z": -2.5, "tick": 12,
                    "yaw": 30.0, "pitch": 2.0, "container": 0,
                    "inv_counts": {"log": 2},
                    "hotbar_ids": (17,) + (0,) * 8,
                    "hotbar_counts": (2,) + (0,) * 8,
                    "hotbar_meta": (0,) * 9,
                }
                self.journal = []
                self.ticks = 0

            def reset_pose(self, yaw, pitch):
                self.obs.update(yaw=float(yaw), pitch=float(pitch))
                self.ticks = 0

            def act(self, **_keys):
                self.ticks += 1
                return self.obs

            def close(self):
                pass

        converge = SimpleNamespace(
            locate=Mock(return_value=(5, 64, -2)),
            phi=Mock(side_effect=(0.1, 0.25)))
        success = Check({"type": "pitch_below", "value": -80})
        task = SimpleNamespace(
            path="/project/task.yaml", name="branch", view="semantic",
            raw={}, journal=True, initial_state=SimpleNamespace(world={}),
            shaping=[], converge=converge, success=success, fail=None,
            max_turns=11, max_ticks_per_turn=23,
        )
        digest = "a" * 64
        state = _state(
            f"datasets/branch_snapshots/{digest}.bsnp", digest)
        env = FakeEnv()

        with patch("bedrock_rl.env.episode.open_view",
                   return_value=(object(), Path("/selected-engine"), {
                       "type": "semantic",
                   })), patch("bedrock_rl.env.episode._open_env",
                              return_value=env), patch(
                                  "bedrock_rl.env.episode.capture_png",
                                  side_effect=lambda _view, episode: (
                                      episode.env.act(snap=1),
                                      b"branch-t0")[1]):
            episode = Episode(
                task, 7, 30, 2, parser=lambda *_args: ([], 0),
                branch_state=state, raster="cpu", view_distance=8,
                render_set={"anim_textures": 0, "no_hand": 1})

        self.assertEqual(episode.spec["branch_state"], state)
        self.assertEqual(episode.spec["runtime"], state["runtime"])
        self.assertEqual(episode.netherite_home, "/selected-engine")
        self.assertEqual(episode.start_obs["inv_counts"], {"log": 2})
        env.obs["inv_counts"]["log"] = 3
        gained = Check({"type": "item_gained", "item": "log", "count": 1})
        self.assertTrue(gained.holds(
            env, episode.start_obs, episode.journal_start))
        self.assertEqual((episode.phi, episode.converged), (0.25, 0.0))
        self.assertEqual(episode.t0_png, b"branch-t0")
        episode.close()

    def test_setup_movement_rejects_branch_before_model_action(self):
        task = SimpleNamespace(
            path="/project/task.yaml", name="branch", view="semantic",
            raw={}, journal=True, initial_state=SimpleNamespace(world={}),
            shaping=[], converge=None,
            success=Check({"type": "pitch_below", "value": -80}), fail=None,
            max_turns=4, max_ticks_per_turn=10,
        )
        obs = {
            "x": 1.0, "y": 64.0, "z": 1.0, "tick": 1,
            "yaw": 0.0, "pitch": 0.0, "container": 0,
            "inv_counts": {}, "hotbar_ids": (0,) * 9,
            "hotbar_counts": (0,) * 9, "hotbar_meta": (0,) * 9,
        }
        env = SimpleNamespace(
            obs=obs, journal=[], ticks=0,
            reset_pose=lambda *_args: None, close=lambda: None)

        def moved(_view, episode):
            episode.env.obs = {**obs, "x": 1.1}
            return b"moved"

        digest = "a" * 64
        state = _state(
            f"datasets/branch_snapshots/{digest}.bsnp", digest)
        with patch("bedrock_rl.env.episode.open_view",
                   return_value=(object(), Path("/engine"), {})), patch(
                       "bedrock_rl.env.episode._open_env",
                       return_value=env), patch(
                           "bedrock_rl.env.episode.capture_png",
                           side_effect=moved), self.assertRaisesRegex(
                               RuntimeError, "setup changed"):
            Episode(task, 7, 0, 0, object(), branch_state=state)

    def test_already_solved_model_visible_t0_is_rejected(self):
        task = SimpleNamespace(
            path="/project/task.yaml", name="branch", view="semantic",
            raw={}, journal=True, initial_state=SimpleNamespace(world={}),
            shaping=[], converge=None,
            success=Check({"type": "pitch_below", "value": 1}), fail=None,
            max_turns=4, max_ticks_per_turn=10,
        )
        obs = {
            "x": 1.0, "y": 64.0, "z": 1.0, "tick": 1,
            "yaw": 0.0, "pitch": 0.0, "container": 0,
            "inv_counts": {}, "hotbar_ids": (0,) * 9,
            "hotbar_counts": (0,) * 9, "hotbar_meta": (0,) * 9,
        }
        env = SimpleNamespace(
            obs=obs, journal=[], ticks=0,
            reset_pose=lambda *_args: None, close=lambda: None)
        digest = "a" * 64
        state = _state(
            f"datasets/branch_snapshots/{digest}.bsnp", digest)
        with patch("bedrock_rl.env.episode.open_view",
                   return_value=(object(), Path("/engine"), {})), patch(
                       "bedrock_rl.env.episode._open_env",
                       return_value=env), patch(
                           "bedrock_rl.env.episode.capture_png",
                           return_value=b"stable"), self.assertRaisesRegex(
                               RuntimeError, "already terminal"):
            Episode(
                task, 7, 0, 0, object(), branch_state=state,
                render_set={"anim_textures": 0, "no_hand": 1})


class BranchPointGeneratorTests(unittest.TestCase):
    @staticmethod
    def _task(path, *, mobs=False, include_time=True):
        world = "mobs: " + ("true" if mobs else "false")
        if include_time:
            world += ", time: 1000"
        path.write_text("\n".join((
            "name: branch_test",
            "goal: Wait.",
            "seeds: [7]",
            "max_turns: 4",
            f"initial_state: {{world: {{{world}}}}}",
            "success: {type: pitch_below, value: -80}",
        )))

    def test_rejects_omitted_snapshot_domains(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.yaml"
            for mobs, time, pattern in (
                    (True, True, "mobs"),
                    (False, False, "world.time")):
                self._task(path, mobs=mobs, include_time=time)
                with self.subTest(mobs=mobs, time=time), \
                        self.assertRaisesRegex(ValueError, pattern):
                    BranchPointDatasetGenerator(
                        path, prefix=[[{"action": "wait"}]],
                        continuation=[[{"action": "wait"}]])

            self._task(path)
            rejected = (
                {"action": "left_click"},
                {"action": "right_click"},
                {"action": "key", "k": "e"},
                {"action": "key", "k": "1"},
            )
            for action in rejected:
                with self.subTest(action=action), self.assertRaises(ValueError):
                    BranchPointDatasetGenerator(
                        path, prefix=[[action]],
                        continuation=[[{"action": "wait"}]])

    def test_producer_rejects_unsupported_runtime_and_settle_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.yaml"
            for declaration, error in (
                    ("raster: cuda", ValueError),
                    ("view_distance: true", TypeError),
                    ("view_distance: 0", ValueError),
                    ("view_distance: 9", ValueError)):
                self._task(path)
                path.write_text(path.read_text() + f"\n{declaration}\n")
                with self.subTest(declaration=declaration), \
                        self.assertRaises(error):
                    BranchPointDatasetGenerator(
                        path, prefix=[[{"action": "wait"}]],
                        continuation=[[{"action": "wait"}]])
            self._task(path)
            for settle_ticks in (1, 201):
                with self.subTest(settle_ticks=settle_ticks), \
                        self.assertRaises(ValueError):
                    BranchPointDatasetGenerator(
                        path, prefix=[[{"action": "wait"}]],
                        continuation=[[{"action": "wait"}]],
                        settle_ticks=settle_ticks)

    def test_penalized_prefix_is_rejected(self):
        episode = SimpleNamespace(
            penalty=0.0,
            step=lambda *_args, **_kwargs: (
                setattr(episode, "penalty", 0.2)
                or ("bad", None, -0.2, False)))
        with self.assertRaisesRegex(Exception, "malformed or penalized"):
            BranchPointDatasetGenerator._checked_step(
                episode, [{"action": "wait"}], "prefix")

    def test_row_uses_branch_home_runtime_and_matching_t0(self):
        frame = _png()
        obs = {"tick": 12, "yaw": 14.0, "pitch": -2.0, "value": 1}

        class FakeEpisode:
            def __init__(self, *, t0=None):
                self.env = SimpleNamespace(
                    obs=dict(obs), engine_tick_origin=10,
                    snapshot=lambda path, radius: Path(path).write_bytes(b"x"))
                self.done = False
                self.penalty = 0.0
                self.netherite_home = "/view-selected-engine"
                self.raster = "cpu"
                self.view_distance = 6
                self.t0_png = frame if t0 is None else t0
                self.spec = {}
                self.close = Mock()

            def step(self, *_args, **_kwargs):
                return ("", frame, 0.0, False)

            def frame(self):
                return frame

        source = FakeEpisode()
        probe = FakeEpisode()
        final = FakeEpisode()
        digest = "a" * 64
        state = _state(
            f"datasets/branch_snapshots/{digest}.bsnp", digest,
            view_distance=6)
        final.spec = {
            **episode_spec("task.yaml", 7, 14, -2, state),
            "view": {"type": "semantic"},
            "runtime": state["runtime"],
        }

        with tempfile.TemporaryDirectory() as directory:
            task_path = Path(directory) / "task.yaml"
            self._task(task_path)
            with patch("bedrock_rl.sdg.pipeline.Episode.draw",
                       return_value=source), patch.object(
                           BranchPointDatasetGenerator, "_write_snapshot",
                           return_value=Path(directory) / f"{digest}.bsnp"), \
                    patch("bedrock_rl.sdg.pipeline.make_branch_state",
                          return_value=state) as make_state, patch(
                              "bedrock_rl.sdg.pipeline.Episode.from_spec",
                              side_effect=(probe, final)) as reopen, patch(
                                  "bedrock_rl.sdg.pipeline.cu_parser",
                                  return_value=object()), patch(
                                      "bedrock_rl.sdg.pipeline.cu_user_prompt",
                                      return_value="<image>\nTASK: Wait."), \
                    patch("bedrock_rl.sdg.pipeline."
                          "inspect_quiescent_snapshot",
                          return_value={"snapshot_tick": 12}):
                generator = BranchPointDatasetGenerator(
                    task_path,
                    prefix=[[{"action": "wait", "ticks": 2}]],
                    continuation=[[{"action": "key", "k": "w", "ticks": 1}]],
                    settle_ticks=3)
                row = generator(Case(0, 7, 11, {}))

        self.assertEqual(row["images"], [{"bytes": frame, "path": None}])
        create = row["extra_info"]["tools_kwargs"]["computer"][
            "create_kwargs"]
        self.assertEqual(create["branch_state"], state)
        self.assertEqual(create["runtime"], state["runtime"])
        self.assertEqual(
            row["extra_info"]["branch_probe"]["snapshot_format"], "BSNP/1")
        self.assertEqual(
            row["extra_info"]["branch_probe"]["scope"],
            "exact diagnostic trace only; zero guaranteed continuation ticks")
        self.assertEqual(row["extra_info"]["branch_contract"], {
            "status": "experimental",
            "training_eligible": False,
            "guaranteed_continuation_ticks": 0,
        })
        self.assertEqual(
            row["extra_info"]["branch_probe"]["continuation_ticks"], 0)
        self.assertEqual(
            row["extra_info"]["branch_probe"]["source_tick_start"], 12)
        self.assertEqual(
            row["extra_info"]["branch_probe"]["source_tick_end"], 12)
        self.assertEqual(
            make_state.call_args.kwargs["netherite_home"],
            "/view-selected-engine")
        self.assertEqual(make_state.call_args.kwargs["world_time"], 1002)
        self.assertEqual(
            reopen.call_args_list[0].kwargs["netherite_home"],
            "/view-selected-engine")
        self.assertEqual(
            reopen.call_args_list[1].kwargs["netherite_home"],
            "/view-selected-engine")
        source.close.assert_called_once_with()
        probe.close.assert_called_once_with()
        final.close.assert_called_once_with()

    def test_actual_parquet_round_trip_preserves_coordinate_and_image(self):
        digest = "a" * 64
        state = _state(f"datasets/branch_snapshots/{digest}.bsnp", digest)
        image = _png((1, 2, 3))
        create = {
            **episode_spec("task.yaml", 7, 0, 0, state),
            "view": {"type": "semantic"},
            "runtime": state["runtime"],
            "interface": "cu",
        }
        row = {
            "data_source": "bedrock",
            "prompt": [{"role": "user", "content": "<image>\nTASK: Wait."}],
            "ability": "game",
            "reward_model": {"style": "rule", "ground_truth": ""},
            "extra_info": {
                **create, "need_tools_kwargs": True,
                "tools_kwargs": {"computer": {"create_kwargs": create}},
            },
            "agent_name": "bedrock",
            "images": [{"bytes": image, "path": None}],
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rows.parquet"
            write_rows([row], output)
            import datasets
            loaded = datasets.load_dataset(
                "parquet", data_files=str(output), split="train")[0]
        loaded_create = loaded["extra_info"]["tools_kwargs"]["computer"][
            "create_kwargs"]
        self.assertEqual(loaded_create["branch_state"], state)
        self.assertEqual(loaded_create["runtime"], state["runtime"])
        self.assertEqual(loaded["images"][0].convert("RGB").getpixel((0, 0)),
                         (1, 2, 3))

    def test_netherite_environment_forwards_fifth_coordinate(self):
        digest = "a" * 64
        state = _state(
            f"datasets/branch_snapshots/{digest}.bsnp", digest)
        task = SimpleNamespace(path="/project/task.yaml")
        opened = SimpleNamespace()
        environment = NetheriteEnvironment(
            "/project/task.yaml",
            episode={"seed": 7, "init_yaw": 1, "init_pitch": 2,
                     "branch_state": state})
        with patch(
                "bedrock_rl.adapters.netherite.environment."
                "NetheriteTask.load", return_value=task), patch(
                    "bedrock_rl.adapters.netherite.environment.cu_parser",
                    return_value=object()), patch(
                        "bedrock_rl.adapters.netherite.environment."
                        "Episode.from_spec", return_value=opened) as replay:
            session = environment._open_blocking("/project/task.yaml")
        self.assertIs(session.episode, opened)
        self.assertEqual(
            replay.call_args.args[0]["branch_state"], state)


if __name__ == "__main__":
    unittest.main()

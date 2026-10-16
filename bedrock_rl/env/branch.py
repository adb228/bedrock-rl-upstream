"""Portable, empirically probed references to Netherite BSNP v1 snapshots.

This is deliberately not a general mid-episode save-state API.  BSNP v1 is
the original twelve-patch format.  It is sufficient for deterministic
initial-state caching, but omits live-simulation and clock state needed for a
general continuation.  A branch row is accepted only after source and restore
match for the exact short probe recorded by the producer.  That finite
measurement is evidence about that row and action sequence, not a guarantee
for arbitrary actions or later ticks.

Task YAML and snapshot paths belong to a trusted local dataset bundle.  Path
validation confines accidental references, but is not a hostile-filesystem
sandbox.  Content TOCTOU is removed: one opened source file is hashed while
being copied to a mode-0400 process-private directory, and the engine opens
that verified copy.
"""

from __future__ import annotations

import copy
import hashlib
import os
import re
import shutil
import stat
import struct
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from collections.abc import Mapping


BRANCH_STATE_SCHEMA = "bedrock.branch-state.v1"
CERTIFICATION_PROFILE = "netherite.bsnp-v1-experimental.v1"
BRANCH_SNAPSHOT_DIR = "datasets/branch_snapshots"
MAX_PROBE_TICKS = 8
BRANCH_STATE_KEYS = frozenset((
    "schema", "snapshot_ref", "snapshot_sha256", "task_sha256",
    "engine_identity", "runtime", "certification",
))
RUNTIME_KEYS = frozenset((
    "raster", "view_distance", "render_set", "random_ticks",
))
CERTIFICATION_KEYS = frozenset((
    "profile", "snapshot_format", "action_domain", "snapshot_tick",
    "world_time", "settle_ticks", "max_probe_ticks",
))
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
APPROVED_PATCH_STACK = tuple(
    f"{number:04d}-{name}.diff" for number, name in enumerate((
        "rl-env", "event-journal", "vanilla-obsidian-speed",
        "complete-crafting", "safe-portal-arrival", "long-episode-drops",
        "sneak-use-placement", "route-contract-test",
        "ten-chunk-snapshots", "emerald-harvest", "redstone-harvest",
        "full-inventory-observation",
    ), 1)
)


def snapshot_sha256(path):
    """SHA-256 of one snapshot without loading it all into memory."""

    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_sha256(path):
    """Digest the exact task-YAML bytes named by an episode."""

    return snapshot_sha256(path)


def _engine_identity(netherite_home=None, raster="cpu"):
    # Imported lazily: snapshots imports Episode's frame dimensions, while
    # Episode imports this module before opening an environment.
    from bedrock_rl.env import engine
    from bedrock_rl.env.snapshots import engine_cache_identity

    binary = engine.find_binary(netherite_home, raster)
    identity = engine_cache_identity(netherite_home)
    actual = identity["actual"]
    expected = identity["expected"]
    manifest = actual.get("manifest", {})
    status = engine.patch_status(netherite_home)
    expected_names = tuple(sorted(expected.get("manifest", {})))
    if (actual.get("head") != expected.get("head")
            or expected_names != APPROVED_PATCH_STACK
            or manifest.get("source") != "recorded"
            or manifest.get("patches") != expected.get("manifest")
            or status["problems"]
            or tuple(sorted(status["adopted"])) != APPROVED_PATCH_STACK):
        raise RuntimeError(
            "BSNP v1 branch rows require the exact pinned Netherite commit "
            "and recorded original 0001-0012 patch stack")
    target = "game" if raster == "cpu" else "game-cuda"
    current = subprocess.run(
        ("make", "-q", "-C", os.fspath(binary.parent), target),
        capture_output=True, text=True)
    if current.returncode != 0:
        detail = (current.stderr or current.stdout).strip()
        raise RuntimeError(
            "branch engine binary is stale for the original patch stack "
            f"({target}; make -q={current.returncode})"
            + (f": {detail}" if detail else ""))
    identity["binary"] = {
        "name": binary.name,
        "sha256": snapshot_sha256(binary),
    }
    identity["approval"] = {
        "exact_original_0012_stack": True,
        "build_target": target,
        "make_q": 0,
    }
    return identity


def _closed(mapping, allowed, what):
    unknown = sorted(set(mapping) - allowed)
    missing = sorted(allowed - set(mapping))
    if unknown:
        raise ValueError(f"unknown {what} keys: " + ", ".join(unknown))
    if missing:
        raise ValueError(f"{what} is missing: " + ", ".join(missing))


def _digest(value, what):
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{what} must be 64 lowercase hex digits")
    return value


def _integer(value, what, *, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{what} must be an integer")
    if value < minimum:
        raise ValueError(f"{what} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{what} must be at most {maximum}")
    return int(value)


def canonical_branch_state(value):
    """Validate and detach the optional fifth episode coordinate."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("episode branch_state must be a mapping")
    _closed(value, BRANCH_STATE_KEYS, "episode branch_state")
    if value["schema"] != BRANCH_STATE_SCHEMA:
        raise ValueError(
            f"unsupported branch_state schema {value['schema']!r}; "
            f"expected {BRANCH_STATE_SCHEMA!r}")

    reference = value["snapshot_ref"]
    if not isinstance(reference, str) or not reference:
        raise TypeError("branch_state snapshot_ref must be a non-empty string")
    pure = PurePosixPath(reference)
    # Equality catches raw duplicate separators and dot segments that
    # PurePosixPath would otherwise silently normalize away.
    if ("\\" in reference or pure.is_absolute()
            or reference != pure.as_posix()
            or any(part in ("", ".", "..") for part in pure.parts)):
        raise ValueError(
            "branch_state snapshot_ref must be canonical task-relative POSIX "
            "syntax without '.', '..', duplicate separators, or backslashes")
    snapshot_digest = _digest(
        value["snapshot_sha256"], "branch_state snapshot_sha256")
    expected_name = f"{snapshot_digest}.bsnp"
    if pure.name != expected_name:
        raise ValueError(
            "branch_state snapshot_ref basename must be "
            f"{expected_name!r}")
    if pure.parent.as_posix() != BRANCH_SNAPSHOT_DIR:
        raise ValueError(
            "branch_state snapshot_ref must use the fixed "
            f"{BRANCH_SNAPSHOT_DIR!r} directory")

    task_digest = _digest(value["task_sha256"], "branch_state task_sha256")
    identity = value["engine_identity"]
    if not isinstance(identity, Mapping):
        raise TypeError("branch_state engine_identity must be a mapping")
    binary = identity.get("binary")
    if not isinstance(binary, Mapping):
        raise ValueError("branch_state engine_identity needs binary identity")
    if set(binary) != {"name", "sha256"}:
        raise ValueError(
            "branch_state engine_identity.binary needs name and sha256")
    _digest(binary["sha256"], "engine binary sha256")

    runtime = value["runtime"]
    if not isinstance(runtime, Mapping):
        raise TypeError("branch_state runtime must be a mapping")
    _closed(runtime, RUNTIME_KEYS, "branch_state runtime")
    if runtime["raster"] != "cpu":
        raise ValueError("experimental branch states require raster 'cpu'")
    view_distance = _integer(
        runtime["view_distance"], "branch_state runtime.view_distance",
        minimum=1, maximum=8)
    render_set = runtime["render_set"]
    if not isinstance(render_set, Mapping):
        raise TypeError("branch_state runtime.render_set must be a mapping")
    if dict(render_set) != {"anim_textures": 0, "no_hand": 1}:
        raise ValueError(
            "experimental branch runtime requires render_set "
            "{'anim_textures': 0, 'no_hand': 1}")
    if runtime["random_ticks"] is not False:
        raise ValueError("BSNP v1 branch runtime requires random_ticks false")

    certification = value["certification"]
    if not isinstance(certification, Mapping):
        raise TypeError("branch_state certification must be a mapping")
    _closed(certification, CERTIFICATION_KEYS, "branch_state certification")
    if certification["profile"] != CERTIFICATION_PROFILE:
        raise ValueError(
            "unsupported branch certification profile "
            f"{certification['profile']!r}")
    if certification["snapshot_format"] != "BSNP/1":
        raise ValueError("branch states require original BSNP/1")
    if certification["action_domain"] != "movement-look-wait":
        raise ValueError(
            "experimental branch action_domain must be 'movement-look-wait'")
    snapshot_tick = _integer(
        certification["snapshot_tick"], "branch snapshot_tick")
    world_time = _integer(certification["world_time"], "branch world_time")
    settle_ticks = _integer(
        certification["settle_ticks"], "branch settle_ticks",
        minimum=2, maximum=200)
    max_probe_ticks = _integer(
        certification["max_probe_ticks"], "branch max_probe_ticks",
        minimum=1, maximum=MAX_PROBE_TICKS)

    return {
        "schema": BRANCH_STATE_SCHEMA,
        "snapshot_ref": pure.as_posix(),
        "snapshot_sha256": snapshot_digest,
        "task_sha256": task_digest,
        "engine_identity": copy.deepcopy(dict(identity)),
        "runtime": {
            "raster": "cpu",
            "view_distance": view_distance,
            "render_set": {"anim_textures": 0, "no_hand": 1},
            "random_ticks": False,
        },
        "certification": {
            "profile": CERTIFICATION_PROFILE,
            "snapshot_format": "BSNP/1",
            "action_domain": "movement-look-wait",
            "snapshot_tick": snapshot_tick,
            "world_time": world_time,
            "settle_ticks": settle_ticks,
            "max_probe_ticks": max_probe_ticks,
        },
    }


def _task_root(task_path):
    if not task_path:
        raise ValueError(
            "a branch_state requires a loaded task YAML path so its portable "
            "snapshot_ref has a controlled root")
    return Path(task_path).expanduser().resolve(strict=True).parent


def branch_output_dir(task_path):
    """The fixed producer-owned artifact directory for branch snapshots."""

    root = _task_root(task_path)
    output = root.joinpath(*PurePosixPath(BRANCH_SNAPSHOT_DIR).parts)
    # Existing symlinks are permitted only when they remain inside the trusted
    # task bundle.  The content-addressed destination itself is never a link.
    if output.exists():
        resolved = output.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                "fixed branch snapshot directory escapes the task bundle") \
                from exc
        return resolved
    return output


def inspect_quiescent_snapshot(path):
    """Read only the quiescence fields that BSNP v1 actually contains."""

    from bedrock_rl.env import snapshots

    data = Path(path).read_bytes()
    snapshots._check_layout(data)
    tick = struct.unpack_from("<q", data, 16)[0]
    motion_off = snapshots.ANGLE_OFF + struct.calcsize("<2f")
    motion = struct.unpack_from("<3d", data, motion_off)
    on_ground, collided_h, collided_v, is_collided = struct.unpack_from(
        "<4i", data, motion_off + struct.calcsize("<3d"))
    fall_distance = struct.unpack_from(
        "<f", data, motion_off + struct.calcsize("<3d4i"))[0]
    sprinting = struct.unpack_from(
        "<i", data, motion_off + struct.calcsize("<3d4if"))[0]
    dig_off = snapshots.VITALS_OFF + struct.calcsize("<fiffi")
    (dig_progress, _hx, _hy, _hz, dig_hitting, dig_delay,
     attack_prev, right_click_delay, use_prev, hurt_velocity_reset) = \
        struct.unpack_from("<f3i6i", data, dig_off)
    container = struct.unpack_from("<i", data, snapshots.INV_OFF - 6 * 4)[0]
    values = {
        "snapshot_tick": int(tick),
        "motion": tuple(float(part) for part in motion),
        "on_ground": int(on_ground),
        "collided_h": int(collided_h),
        "collided_v": int(collided_v),
        "is_collided": int(is_collided),
        "fall_distance": float(fall_distance),
        "sprinting": int(sprinting),
        "dig_progress": float(dig_progress),
        "dig_hitting": int(dig_hitting),
        "dig_delay": int(dig_delay),
        "attack_prev": int(attack_prev),
        "right_click_delay": int(right_click_delay),
        "use_prev": int(use_prev),
        "hurt_velocity_reset": int(hurt_velocity_reset),
        "container": int(container),
        "snapshot_format": "BSNP/1",
    }
    offenders = []
    if values["on_ground"] != 1:
        offenders.append("player is not on ground")
    grounded_gravity = -0.0784000015258789
    if (abs(motion[0]) > 1e-12 or abs(motion[2]) > 1e-12
            or min(abs(motion[1]), abs(motion[1] - grounded_gravity))
            > 1e-9):
        offenders.append(f"player motion is {motion!r}")
    if abs(values["fall_distance"]) > 1e-7:
        offenders.append(f"fall distance is {values['fall_distance']!r}")
    if values["sprinting"]:
        offenders.append("sprinting latch is active")
    if values["container"]:
        offenders.append(f"container {values['container']} is open")
    for key in ("dig_progress", "dig_hitting", "dig_delay", "attack_prev",
                "right_click_delay", "use_prev", "hurt_velocity_reset"):
        if values[key]:
            offenders.append(f"{key} is active ({values[key]!r})")
    if offenders:
        raise ValueError(
            "snapshot is not an eligible quiescent diagnostic branch point: "
            + "; ".join(offenders))
    return values


def make_branch_state(snapshot_path, task_path, *, netherite_home,
                      raster, view_distance, world_time, settle_ticks):
    """Build a measured BSNP v1 fifth coordinate beneath the task bundle."""

    root = _task_root(task_path)
    path = Path(snapshot_path).expanduser().resolve(strict=True)
    try:
        reference = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "branch snapshots must live beneath the task YAML directory") \
            from exc
    digest = snapshot_sha256(path)
    if path.name != f"{digest}.bsnp":
        raise ValueError(
            "branch snapshot filename must be its SHA-256 plus .bsnp")
    inspected = inspect_quiescent_snapshot(path)
    return canonical_branch_state({
        "schema": BRANCH_STATE_SCHEMA,
        "snapshot_ref": PurePosixPath(*reference.parts).as_posix(),
        "snapshot_sha256": digest,
        "task_sha256": task_sha256(task_path),
        "engine_identity": _engine_identity(netherite_home, raster),
        "runtime": {
            "raster": str(raster),
            "view_distance": int(view_distance),
            "render_set": {"anim_textures": 0, "no_hand": 1},
            "random_ticks": False,
        },
        "certification": {
            "profile": CERTIFICATION_PROFILE,
            "snapshot_format": inspected["snapshot_format"],
            "action_domain": "movement-look-wait",
            "snapshot_tick": inspected["snapshot_tick"],
            "world_time": int(world_time),
            "settle_ticks": int(settle_ticks),
            "max_probe_ticks": MAX_PROBE_TICKS,
        },
    })


def _validated_source(task_path, state, *, netherite_home, raster,
                      view_distance, render_set=None, random_ticks=None):
    from bedrock_rl.env.task import Task

    root = _task_root(task_path)
    task = Task.load(os.fspath(task_path))
    world = task.initial_state.world
    if world.get("mobs") is not False or "time" not in world:
        raise RuntimeError(
            "BSNP v1 branch consumer requires task YAML with explicit "
            "initial_state.world.time and initial_state.world.mobs: false")
    if task_sha256(task_path) != state["task_sha256"]:
        raise RuntimeError(
            "branch snapshot task digest is stale: task YAML bytes changed")
    if raster != state["runtime"]["raster"]:
        raise RuntimeError(
            f"branch snapshot requires raster {state['runtime']['raster']!r}, "
            f"not {raster!r}")
    if int(view_distance) != state["runtime"]["view_distance"]:
        raise RuntimeError(
            "branch snapshot view_distance does not match the recorded "
            "runtime")
    if dict(render_set or {}) != state["runtime"]["render_set"]:
        raise RuntimeError(
            "branch snapshot render_set does not match the recorded runtime")
    if random_ticks is not False:
        raise RuntimeError("BSNP v1 branch runtime requires random_ticks false")
    actual_identity = _engine_identity(netherite_home, raster)
    if state["engine_identity"] != actual_identity:
        raise RuntimeError(
            "branch snapshot engine identity is stale: checkout, original "
            "0001-0012 patch stack, or executable bytes do not match")

    pure = PurePosixPath(state["snapshot_ref"])
    candidate = root.joinpath(*pure.parts)
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"branch snapshot {state['snapshot_ref']!r} does not exist "
            f"beneath task directory {root}") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "branch_state snapshot_ref resolves outside the task directory") \
            from exc
    if resolved != candidate:
        raise ValueError("branch snapshot file must not be a symlink")
    return resolved


@contextmanager
def materialized_branch_snapshot(task_path, value, *, netherite_home,
                                 raster, view_distance, render_set=None,
                                 random_ticks=False):
    """Yield a verified private copy plus its canonical coordinate."""

    state = canonical_branch_state(value)
    source = _validated_source(
        task_path, state, netherite_home=netherite_home, raster=raster,
        view_distance=view_distance, render_set=render_set,
        random_ticks=random_ticks)
    directory = Path(tempfile.mkdtemp(prefix="bedrock-branch-open-"))
    os.chmod(directory, 0o700)
    private = directory / f"{state['snapshot_sha256']}.bsnp"
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(source, flags)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("branch snapshot is not a regular file")
            with os.fdopen(fd, "rb", closefd=False) as incoming, \
                    open(private, "xb") as outgoing:
                for chunk in iter(lambda: incoming.read(1 << 20), b""):
                    digest.update(chunk)
                    outgoing.write(chunk)
                outgoing.flush()
                os.fsync(outgoing.fileno())
        finally:
            os.close(fd)
        if digest.hexdigest() != state["snapshot_sha256"]:
            raise RuntimeError(
                "branch snapshot SHA-256 mismatch: recorded "
                f"{state['snapshot_sha256']}, found {digest.hexdigest()}")
        inspected = inspect_quiescent_snapshot(private)
        if inspected["snapshot_tick"] != \
                state["certification"]["snapshot_tick"]:
            raise RuntimeError(
                "branch snapshot tick does not match its certification")
        os.chmod(private, 0o400)
        yield os.fspath(private), state
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def resolve_branch_snapshot(task_path, value, *, netherite_home,
                            raster="cpu", view_distance=8, render_set=None,
                            random_ticks=False):
    """Compatibility validator returning the trusted source path.

    Engine callers must use :func:`materialized_branch_snapshot`; this helper
    exists for diagnostics and tests that do not launch a process.
    """

    state = canonical_branch_state(value)
    source = _validated_source(
        task_path, state, netherite_home=netherite_home, raster=raster,
        view_distance=view_distance, render_set=render_set,
        random_ticks=random_ticks)
    if snapshot_sha256(source) != state["snapshot_sha256"]:
        raise RuntimeError("branch snapshot SHA-256 mismatch")
    inspected = inspect_quiescent_snapshot(source)
    if inspected["snapshot_tick"] != state["certification"]["snapshot_tick"]:
        raise RuntimeError("branch snapshot tick does not match certification")
    return os.fspath(source)

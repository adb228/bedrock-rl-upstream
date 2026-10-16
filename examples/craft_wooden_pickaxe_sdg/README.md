# Craft a wooden pickaxe from an empty inventory

This bounded CPU example validates a complete survival controller chain on one
known world. Seed `0` and decision seed `2026091501` are explicit, the sampler
has one candidate, and `SerialExecutor` preserves a single deterministic
execution order.

The `logs` stage mines three natural logs, then performs these operations in
order:

1. craft at least 12 planks in the 2x2 inventory grid;
2. craft at least 2 sticks in the 2x2 grid;
3. craft one crafting table in the 2x2 grid;
4. place and live-verify that workstation;
5. craft at least one wooden pickaxe in its 3x3 grid.

Twelve planks are declared because three vanilla plank crafts consume the
three logs and produce four planks each. One stick craft consumes two planks
and batches four sticks even though the explicit milestone is the two needed
by the pickaxe. The table consumes four planks, and the pickaxe consumes three
planks plus two sticks:

| Item | Produced | Consumed | Conservative remainder |
| --- | ---: | ---: | ---: |
| log | 3 | 3 | 0 |
| planks | 12 | 2 (sticks) + 4 (table) + 3 (pickaxe) | 3 |
| stick | 4 batched (minimum 2) | 2 | 2 |
| crafting table | 1 | 1 placed | 0 in inventory |
| wooden pickaxe | 1 | 0 | 1 |

Success is only the live task check `item_gained: wooden_pickaxe`. The
trajectory's terminal `after_tools` state includes the exact 36-slot inventory
arrays and aggregate `inv_counts`, so the artifact contains inventory evidence
for the verifier result.

## Validate and run

```bash
uv run bedrock task validate \
  examples/craft_wooden_pickaxe_sdg/task.yaml
uv run bedrock generate \
  examples/craft_wooden_pickaxe_sdg/data.yaml --check
NETHERITE_HOME=/workspace/runtime/netherite uv run bedrock generate \
  examples/craft_wooden_pickaxe_sdg/data.yaml
```

`TrajectoryArtifactWriter` saves the canonical action trajectory and strictly
replays the same actions in the procedural CPU renderer. Replay checks the
terminal flags, reward, turn/action counts, and complete journal event stream
before writing the GIF. Override `output`, `rejections_output`, and
`writer.replay.gif` with `--set` when evidence belongs outside the checkout.

## Scope and grounding

Crafting, workstation placement, and the snapshot-selected route are
privileged/non-grounded turns. This example deliberately writes a trajectory
artifact, not SFT or RL data: it validates the scripted controller, live
environment transitions, recipe execution, and task verifier only. It has no
`train.yaml`, RL prompts, or model configuration, and its metadata marks the
artifact `training_eligible: false`.

The task also satisfies Patch A's diagnostic constructor requirements:
explicit CPU raster, bounded view distance, fixed world time, and `mobs:
false`. That permits separately configured experimental BSNP v1 branch-row
diagnostics. It does not make this artifact a branch row, and Patch A rows
remain `status: experimental`, `training_eligible: false`, with zero
guaranteed continuation ticks.

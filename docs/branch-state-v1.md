# Experimental BSNP v1 branch-coordinate evidence

This experimental branch-coordinate and measurement mechanism uses Bedrock-RL
`0553cbb29a0b60de63c3d7e7ac2a07eba14d0a72` and the original pinned
Netherite stack `0001` through `0012`. It does not require patch `0013`, LIV2,
or a v2 profile.

## What v1 can and cannot restore

BSNP v1 is sufficient for deterministic initial-state caching. It contains
the seed/tick, player and represented controller state, inventory, ordinary
item entities, captured block region, and derived coal index.

It is not a general mid-episode continuation format. Auditing the v1 writer
and loader proves that it omits:

- plant coordinates, age, active flag, RNG, and growth state;
- scheduled falling-block updates and pending landings;
- overflow stacks, positions, delays, and spawn-failure count;
- live-simulation tick and non-item live entity state;
- total/world clocks, daylight state, weather timers and weather RNG;
- mobs/dragon, projectiles, active fluids, portal/dimension and terminal
  state;
- furnace, chest, crafting-grid and cursor state;
- held use/eat/bow and additional controller/render histories.

There is no v1 unsafe-state bitmask or hidden-state digest with which a
consumer could prove those fields absent.

## Delayed-divergence measurement

A diagnostic source fixture seeded one overflow stack, plant RNG/growth, and
a delayed falling update before taking a branch. An instrumented source
retained that known hidden state; the original v1 restore received the
projected BSNP v1 bytes and the same world-clock setting.

The branch header tick was 0. At tick 2:

- complete JSON observations differed in `blocks`, `cam`, `depth`, and
  `edge`;
- the rendered CPU PPMs differed: source
  `a48d0a35cecb7139bf69d33fcf61e9b95a62185b6bc1e80562e79f5f18f43482`
  versus restored
  `748d6ae9c50d79d6477df53fae0c1c4994904926310de7c26c1136016cf7d6e7`;
- projected source and restored v1 snapshots differed;
- projected source region SHA-256 was
  `c79d6a92ec54f55898940ec0901ae8125d91b43a36cbab4b2f14816c594f8ed0`;
- restored v1 region SHA-256 was
  `32cf516647c4e4b63010831ac2719775311fffd416ba1ee9d3f6c7fa06f631bb`.

This instrumented oracle was used only to expose a known omission. The result
disproves any positive universal v1 continuation window.

## Exact farm diagnostic

`tools/validate_branch_v1.py` runs `farm_one_log_sdg` seed 0 through:

1. public look/movement prefix;
2. quiescent wait and content-addressed BSNP v1 snapshot;
3. branch-row generation;
4. row `tools_kwargs` reopen through `Episode.from_spec`;
5. a 4-tick source/restore WAIT comparison, restored replay, and an
   alternate look branch.

The exact WAIT continuation consumed 4 engine ticks from model-visible t0. Two
independent reopens had identical complete observations and CPU PNGs at t0
and after that replay. The alternate action changed both observation and
frame in 8 ticks. Final source/restored BSNP v1 bytes were identical, with
SHA-256
`5093d68eabe8853199c681d9441690057877a65e72befee3d3e7009012eed951`.
The branch snapshot SHA-256 was
`9ee19af6635cc03d78e4695419312b46139a58e2dc3a261f94c68d46e69176e1`
at snapshot tick 41 and aligned world time 1025; the row image SHA-256 was
`954761883298709ba4ae52f9373c7606149dc0a8fc3517980c3abe43c5d03a10`.

The exact WAIT source/restore trace passed for 4 engine ticks. The alternate
look result at 8 ticks is expected different-action branch divergence, not
source/restore equality. The implementation caps diagnostic execution at 8
ticks only to bound measurement cost. It does not establish validity through
8 ticks. General continuation validity is unestablished and the guaranteed
window is zero.

The fifth coordinate binds task and snapshot SHA-256, a controlled
task-relative content-addressed path, pinned checkout and exact recorded
`0001`-`0012` patch identities, selected executable bytes, disabled random
ticks, raster/view/render settings, snapshot tick, aligned boot clock, and the
diagnostic measurement metadata. Generated rows are marked experimental and
training-ineligible; the current verl agent loop rejects them before opening
an environment. Direct `NetheriteEnvironment` diagnostic opening remains
available.
Consumers recheck those values while copying one no-follow source descriptor
to a private mode-0400 file before opening Netherite.

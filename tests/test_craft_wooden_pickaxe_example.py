import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import yaml

from bedrock_rl.env.task import Task
from bedrock_rl.sdg.generation import GenerationSpec
from bedrock_rl.sdg.pipeline import BranchPointDatasetGenerator
from bedrock_rl.sdg.policy import PolicyVeto
from bedrock_rl.sdg.policy.progression import (
    CraftOperation,
    SurvivalProgressionPolicy,
    SurvivalProgressionSpec,
    WorkstationOperation,
    WorkstationSite,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "craft_wooden_pickaxe_sdg"


class CraftWoodenPickaxeExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = yaml.safe_load((EXAMPLE / "data.yaml").read_text())
        cls.task = Task.load(EXAMPLE / "task.yaml")
        cls.generation = GenerationSpec.load(EXAMPLE / "data.yaml")
        cls.progression = SurvivalProgressionSpec.parse(
            cls.raw["generator"]["policy"]["spec"])

    def test_task_success_is_only_terminal_inventory_gain(self):
        self.assertEqual(self.task.raw["success"], {
            "type": "item_gained",
            "item": "wooden_pickaxe",
            "count": 1,
            "reward": 1.0,
        })
        self.assertEqual(self.task.success.type, "item_gained")
        self.assertEqual(self.task.success.spec["item"], "wooden_pickaxe")
        self.assertEqual(self.task.initial_state.inventory, ())

    def test_inline_progression_has_the_exact_logs_recipe_chain(self):
        self.assertEqual(len(self.progression.stages), 1)
        stage = self.progression.stages[0]
        self.assertEqual(stage.name, "logs")
        self.assertEqual(stage.count, 3)
        self.assertEqual(stage.blocks, ("log",))
        self.assertEqual(stage.drop_item, "log")
        self.assertEqual(stage.route_kinds, ("surface",))
        self.assertFalse(stage.natural_only)
        self.assertEqual(
            [operation.to_dict() for operation in stage.after],
            [
                {"type": "craft", "item": "planks",
                 "minimum": 12, "grid": 2},
                {"type": "craft", "item": "stick",
                 "minimum": 2, "grid": 2},
                {"type": "craft", "item": "crafting_table",
                 "minimum": 1, "grid": 2},
                {"type": "workstation", "key": "table_logs",
                 "item": "crafting_table", "block": "crafting_table",
                 "container": "crafting_table"},
                {"type": "craft", "item": "wooden_pickaxe",
                 "minimum": 1, "grid": 3, "table": "table_logs"},
            ],
        )

    def test_recipe_material_budget_covers_every_intermediate(self):
        policy = SurvivalProgressionPolicy(spec=self.progression)
        expected = {
            "log": (3, 3, 0),
            "planks": (12, 9, 3),
            # The ledger is conservative: the declared minimum is two,
            # while the live vanilla recipe batches four sticks.
            "stick": (2, 2, 0),
            "crafting_table": (1, 1, 0),
            "wooden_pickaxe": (1, 0, 1),
        }
        for item, values in expected.items():
            with self.subTest(item=item):
                budget = policy._material_budget(item)
                self.assertEqual(
                    (budget["produced"],
                     budget["consumed_before_structure"],
                     budget["available"]),
                    values,
                )

    @staticmethod
    def _empty_environment():
        return SimpleNamespace(obs={
            "x": 0.5,
            "z": 0.5,
            "container": 0,
            "inv_counts": {},
            "inventory_ids": [0] * 36,
            "inventory_counts": [0] * 36,
            "inventory_meta": [0] * 36,
        })

    def test_missing_recipe_intermediate_fails_closed(self):
        policy = SurvivalProgressionPolicy(spec=self.progression)
        policy._stage_index = 0
        policy._operation_index = 1
        policy.item_count = Mock(return_value=0)
        policy._craft_hotbar_swaps = Mock(return_value=())
        policy.observed_inventory_stack = Mock(return_value=None)
        policy.prepare_recipe_ingredients = Mock(return_value=None)
        policy.craft_2x2 = Mock(return_value=None)

        result = policy._operation_actions(
            self._empty_environment(), CraftOperation("stick", 2, 2))

        self.assertIsInstance(result, PolicyVeto)
        self.assertEqual(result.code, "progression_recipe_unavailable")

    def test_missing_placed_workstation_fails_before_3x3_click(self):
        policy = SurvivalProgressionPolicy(spec=self.progression)
        policy._stage_index = 0
        policy._operation_index = 4
        policy.item_count = Mock(return_value=0)
        policy._craft_hotbar_swaps = Mock(return_value=())
        policy.observed_inventory_stack = Mock(return_value=None)
        policy.prepare_recipe_ingredients = Mock(return_value=None)
        policy._workstation_ready = Mock(return_value=False)
        policy.craft_3x3 = Mock()
        policy.workstation_sites["table_logs"] = WorkstationSite(
            "table_logs", (1, 64, 0), (1, 63, 0), (0, 64, 0))

        result = policy._operation_actions(
            self._empty_environment(),
            CraftOperation("wooden_pickaxe", 1, 3, "table_logs"),
        )

        self.assertIsInstance(result, PolicyVeto)
        self.assertEqual(result.code, "progression_crafting_table_missing")
        policy.craft_3x3.assert_not_called()

    def test_manifest_is_one_cpu_artifact_case_and_not_training(self):
        self.assertEqual(self.generation.worlds, 1)
        self.assertEqual(self.generation.max_world_attempts, 1)
        self.assertEqual(
            self.generation.sampler.type, "bedrock_rl.sdg:SeedCases")
        self.assertEqual(self.generation.sampler.config["seeds"], [0])
        self.assertEqual(
            self.generation.sampler.config["decision_seeds"], [2026091501])
        self.assertEqual(
            self.generation.executor.type, "bedrock_rl.sdg:SerialExecutor")
        self.assertEqual(
            self.generation.writer.type,
            "bedrock_rl.sdg:TrajectoryArtifactWriter")
        self.assertFalse(self.generation.metadata["training_eligible"])
        self.assertNotIn(
            "TrajectorySFTWriter", (EXAMPLE / "data.yaml").read_text())
        self.assertFalse((EXAMPLE / "train.yaml").exists())
        self.assertEqual(
            sorted(path.name for path in EXAMPLE.iterdir() if path.is_file()),
            ["README.md", "data.yaml", "task.yaml"],
        )

    def test_patch_a_compatibility_does_not_claim_training_eligibility(self):
        generator = BranchPointDatasetGenerator(
            task=EXAMPLE / "task.yaml",
            prefix=[[{"action": "wait", "ticks": 1}]],
            continuation=[[{"action": "wait", "ticks": 1}]],
            settle_ticks=2,
        )
        self.assertEqual(generator.task.raw["raster"], "cpu")
        self.assertEqual(generator.task.raw["view_distance"], 8)
        self.assertIs(generator.task.initial_state.world["mobs"], False)
        self.assertEqual(generator.task.initial_state.world["time"], 1000)
        self.assertFalse(self.generation.metadata["training_eligible"])
        self.assertNotEqual(
            self.raw["generator"]["type"],
            "bedrock_rl.sdg:BranchPointDatasetGenerator",
        )

    def test_readme_declares_privileged_non_grounded_scope(self):
        text = (EXAMPLE / "README.md").read_text()
        self.assertIn("privileged/non-grounded", text)
        self.assertIn("controller, live", text)
        self.assertIn("training_eligible: false", text)
        self.assertIn("guaranteed continuation ticks", text)


if __name__ == "__main__":
    unittest.main()

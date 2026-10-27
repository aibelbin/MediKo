import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from compare_models import best_epoch, config_differences, format_table, load_result  # noqa: E402
from models import MODELS  # noqa: E402

HISTORY = [
    {"loss": 0.5, "step": 10, "epoch": 0.5},
    {"eval_f1": 0.80, "epoch": 1.0},
    {"eval_f1": 0.86, "epoch": 2.0},
    {"eval_f1": 0.84, "epoch": 3.0},
    {"train_runtime": 600.0, "epoch": 3.0},
    {"validation_f1": 0.86, "epoch": 3.0},
]


def write_run(output_dir: Path, key: str, config: dict) -> None:
    metrics = output_dir / key / "metrics"
    metrics.mkdir(parents=True)
    files = {
        "validation.json": {"validation_precision": 0.85, "validation_recall": 0.87, "validation_f1": 0.86},
        "test.json": {"test_precision": 0.84, "test_recall": 0.88, "test_f1": 0.86},
        "train_config.json": {"model": key, "model_id": MODELS[key]["id"], **config},
        "log_history.json": HISTORY,
    }
    for name, content in files.items():
        (metrics / name).write_text(json.dumps(content))


class ModelsTest(unittest.TestCase):
    def test_biobert_tokenizer_keeps_case(self):
        # Its repo has no tokenizer config, so the default would lowercase a cased vocabulary.
        self.assertEqual(MODELS["biobert"]["tokenizer_kwargs"], {"do_lower_case": False})


class CompareModelsTest(unittest.TestCase):
    def test_best_epoch_is_highest_validation_f1(self):
        self.assertEqual(best_epoch(HISTORY), 2.0)
        self.assertIsNone(best_epoch([{"loss": 0.5}]))

    def test_load_result_reads_metrics_and_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_run(Path(tmp), "biobert", {"epochs": 3})
            result = load_result("biobert", Path(tmp))
            self.assertEqual(result["name"], "BioBERT")
            self.assertEqual(result["test_f1"], 0.86)
            self.assertEqual(result["best_epoch"], 2.0)
            self.assertEqual(result["train_minutes"], 10.0)
            self.assertIsNone(load_result("bert", Path(tmp)))

    def test_config_differences_ignore_the_model_itself(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_run(Path(tmp), "bert", {"epochs": 3, "lr": 2e-5})
            write_run(Path(tmp), "biobert", {"epochs": 3, "lr": 2e-5})
            write_run(Path(tmp), "biomedbert", {"epochs": 1, "lr": 2e-5})
            same = [load_result(k, Path(tmp)) for k in ("bert", "biobert")]
            self.assertEqual(config_differences(same), [])
            mixed = [load_result(k, Path(tmp)) for k in MODELS]
            self.assertEqual(config_differences(mixed), ["epochs"])

    def test_table_has_a_row_per_model_and_dashes_for_missing_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_run(Path(tmp), "bert", {})
            result = load_result("bert", Path(tmp))
            result["train_minutes"] = None
            lines = format_table([result]).splitlines()
            self.assertEqual(len(lines), 3)
            self.assertTrue(lines[2].startswith("BERT "))
            self.assertIn("0.8600", lines[2])
            self.assertTrue(lines[2].rstrip().endswith("-"))


if __name__ == "__main__":
    unittest.main()

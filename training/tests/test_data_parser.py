import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prepare_data import IGNORE_INDEX, LABEL2ID, align_labels, read_conll  # noqa: E402


def write_tmp(content: str) -> Path:
    f = tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False, encoding="utf-8")
    f.write(content)
    f.close()
    return Path(f.name)


class ReadConllTest(unittest.TestCase):
    def test_parses_examples_and_final_example_without_trailing_blank(self):
        path = write_tmp("The\tO\nbreast\tB-Disease\ncancer\tI-Disease\n\n\nIt\tO\nends\tO")
        examples = read_conll(path)
        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0]["tokens"], ["The", "breast", "cancer"])
        self.assertEqual(examples[0]["ner_tags"], [0, LABEL2ID["B-Disease"], LABEL2ID["I-Disease"]])
        self.assertEqual(examples[1]["tokens"], ["It", "ends"])

    def test_rejects_malformed_rows(self):
        with self.assertRaises(ValueError):
            read_conll(write_tmp("token-without-tag\n"))
        with self.assertRaises(ValueError):
            read_conll(write_tmp("word\tB-Gene\n"))

    def test_rejects_empty_file(self):
        with self.assertRaises(ValueError):
            read_conll(write_tmp("\n\n"))


class AlignLabelsTest(unittest.TestCase):
    def test_first_subword_labelled_rest_ignored(self):
        # [CLS] adeno ##matous polyposis [SEP]
        word_ids = [None, 0, 0, 1, None]
        labels = align_labels(word_ids, [1, 2])
        self.assertEqual(labels, [IGNORE_INDEX, 1, IGNORE_INDEX, 2, IGNORE_INDEX])


if __name__ == "__main__":
    unittest.main()

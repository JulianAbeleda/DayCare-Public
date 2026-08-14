import json
from pathlib import Path
import tempfile
import unittest

from daycare.image import ImageProject


class ImageProjectTests(unittest.TestCase):
    def test_create_is_generic_and_outside_trainer_implementation(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = ImageProject.create(
                Path(temporary) / "ink-style",
                name="ink-style",
                trigger="dcstyle_ink",
                base_model="example/base-model",
            )
            self.assertEqual(project.config["kind"], "image-style-lora")
            self.assertEqual(project.config["trainer"]["backend"], "external")
            self.assertNotIn("visual novel", json.dumps(project.config).lower())

    def test_validation_requires_caption_pairs_and_holdout(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = ImageProject.create(
                Path(temporary) / "test-style",
                name="test-style",
                trigger="dcstyle_test",
                base_model="example/base-model",
            )
            project.config["minimum_train_images"] = 1
            (project.root / "project.json").write_text(json.dumps(project.config), encoding="utf-8")
            (project.root / "data/train/example.png").write_bytes(b"not decoded by validator")
            report = ImageProject.load(project.root).validate()
            self.assertFalse(report.ok)
            self.assertIn("missing caption for data/train/example.png", report.errors)
            self.assertIn("holdout split must contain at least one image", report.errors)

    def test_validation_accepts_captioned_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = ImageProject.create(
                Path(temporary) / "test-style",
                name="test-style",
                trigger="dcstyle_test",
                base_model="example/base-model",
            )
            project.config["minimum_train_images"] = 1
            (project.root / "project.json").write_text(json.dumps(project.config), encoding="utf-8")
            for split in ("train", "holdout"):
                (project.root / f"data/{split}/example.png").write_bytes(b"image")
                (project.root / f"data/{split}/example.txt").write_text("1girl, standing", encoding="utf-8")
            self.assertTrue(ImageProject.load(project.root).validate().ok)


if __name__ == "__main__":
    unittest.main()

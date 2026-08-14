"""Filesystem contract for reproducible, trainer-neutral image-style projects."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re


SCHEMA_VERSION = 1
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True)
class ValidationReport:
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    train_images: int
    holdout_images: int

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class ImageProject:
    root: Path
    config: dict

    @classmethod
    def create(cls, root: Path, *, name: str, trigger: str, base_model: str) -> "ImageProject":
        root = root.expanduser().resolve()
        if root.exists() and any(root.iterdir()):
            raise ValueError(f"project directory is not empty: {root}")
        if not SLUG.fullmatch(name):
            raise ValueError("name must be a lowercase kebab-case slug")
        if not trigger.strip() or any(character.isspace() for character in trigger):
            raise ValueError("trigger must be one non-empty token")
        if not base_model.strip():
            raise ValueError("base model is required")

        for relative in ("data/train", "data/holdout", "samples", "evaluations", "runs"):
            (root / relative).mkdir(parents=True, exist_ok=True)
        config = {
            "schema_version": SCHEMA_VERSION,
            "kind": "image-style-lora",
            "name": name,
            "trigger": trigger,
            "base_model": base_model,
            "caption_extension": ".txt",
            "minimum_train_images": 20,
            "trainer": {"backend": "external", "config": "training.toml"},
        }
        (root / "project.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        (root / "training.toml").write_text(
            "# Trainer-specific settings belong here. DayCare keeps the dataset contract generic.\n",
            encoding="utf-8",
        )
        (root / "README.md").write_text(
            f"# {name}\n\nGeneric image-style LoRA project managed by DayCare.\n",
            encoding="utf-8",
        )
        return cls(root, config)

    @classmethod
    def load(cls, root: Path) -> "ImageProject":
        root = root.expanduser().resolve()
        config = json.loads((root / "project.json").read_text(encoding="utf-8"))
        if config.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {config.get('schema_version')!r}")
        if config.get("kind") != "image-style-lora":
            raise ValueError(f"unsupported project kind: {config.get('kind')!r}")
        return cls(root, config)

    def validate(self) -> ValidationReport:
        errors: list[str] = []
        warnings: list[str] = []
        caption_extension = self.config.get("caption_extension", ".txt")

        def inspect(split: str) -> int:
            directory = self.root / "data" / split
            if not directory.is_dir():
                errors.append(f"missing data/{split} directory")
                return 0
            images = sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
            for image in images:
                caption = image.with_suffix(caption_extension)
                if not caption.is_file():
                    errors.append(f"missing caption for data/{split}/{image.name}")
                elif not caption.read_text(encoding="utf-8").strip():
                    errors.append(f"empty caption for data/{split}/{image.name}")
            captions = {path for path in directory.glob(f"*{caption_extension}")}
            expected = {image.with_suffix(caption_extension) for image in images}
            for orphan in sorted(captions - expected):
                warnings.append(f"orphan caption: data/{split}/{orphan.name}")
            return len(images)

        train_images = inspect("train")
        holdout_images = inspect("holdout")
        minimum = int(self.config.get("minimum_train_images", 20))
        if train_images < minimum:
            errors.append(f"train split has {train_images} images; minimum is {minimum}")
        if holdout_images == 0:
            errors.append("holdout split must contain at least one image")
        if self.config.get("trigger", "") == "":
            errors.append("trigger is required")
        if self.config.get("base_model", "") == "":
            errors.append("base_model is required")
        return ValidationReport(tuple(errors), tuple(warnings), train_images, holdout_images)

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from tools.annotate_button_state import AnnotationApp, AnnotationStore, load_candidates, render_candidate, source_group


def make_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    for split in ("train", "valid"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
    image = root / "train" / "images" / "42_jpg.rf.abc.jpg"
    Image.new("RGB", (100, 80), "gray").save(image)
    (root / "train" / "labels" / "42_jpg.rf.abc.txt").write_text(
        "0 0.5 0.5 0.4 0.5\n1 0.2 0.2 0.1 0.1\n", encoding="utf-8"
    )
    data = tmp_path / "dataset.yaml"
    data.write_text(
        f"path: {root}\ntrain: train/images\nval: valid/images\nnames: ['3', 'led']\n", encoding="utf-8"
    )
    return data


def test_load_filters_components_and_groups_sources(tmp_path: Path) -> None:
    candidates = load_candidates(make_dataset(tmp_path), ("train",))
    assert len(candidates) == 1
    assert candidates[0].label == "3"
    assert candidates[0].source_group == "42_jpg"
    assert source_group("plain.jpg") == "plain"


def test_store_saves_and_undoes(tmp_path: Path) -> None:
    data = make_dataset(tmp_path)
    candidate = load_candidates(data, ("train",))[0]
    output = tmp_path / "annotations.json"
    store = AnnotationStore(output, data)
    store.set(candidate, "ON")
    assert store.state_for(candidate.key) == "ON"
    assert json.loads(output.read_text())["annotations"][candidate.key]["state"] == "ON"
    assert store.undo() == candidate.key
    assert store.state_for(candidate.key) is None


def test_navigation_and_rendering(tmp_path: Path) -> None:
    data = make_dataset(tmp_path)
    candidates = load_candidates(data, ("train",))
    store = AnnotationStore(tmp_path / "annotations.json", data)
    app = AnnotationApp(candidates, store)
    assert app.navigate(None, 1, True, "") == candidates[0]
    store.set(candidates[0], "OFF")
    assert app.navigate(None, 1, True, "") is None
    assert app.navigate(None, 1, False, "") == candidates[0]
    assert render_candidate(candidates[0], "crop").startswith(b"\xff\xd8")

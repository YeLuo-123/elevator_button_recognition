import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "test_hybrid_button_state_video.py"
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location("test_hybrid_button_state_video", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
deduplicate = module.deduplicate
propagate_states = module.propagate_states


def test_brightness_baseline_holds_on_until_confirmed_fall():
    rows = [
        {"time_s": 0.0, "brightness_feature": 40.0, "qwen_state": "OFF", "review_reasons": []},
        {"time_s": 0.2, "brightness_feature": 65.0, "qwen_state": "ON", "review_reasons": ["brightness_rise"]},
        {"time_s": 1.8, "brightness_feature": 70.0, "qwen_state": "OFF", "review_reasons": ["periodic"]},
        {"time_s": 3.0, "brightness_feature": 42.0, "qwen_state": "OFF", "review_reasons": ["brightness_fall"]},
        {"time_s": 3.25, "brightness_feature": 41.0, "qwen_state": "OFF", "review_reasons": ["brightness_followup"]},
    ]
    propagate_states({1: SimpleNamespace(rows=rows)}, off_delay=1.2, fast_off_delay=.2,
                     brightness_hold_margin=8.0)
    assert [row["state"] for row in rows] == ["OFF", "ON", "ON", "ON", "OFF"]


def test_deduplicate_keeps_highest_confidence_overlapping_box():
    rows = [
        {"bbox": [0, 0, 10, 10], "detection_confidence": .8},
        {"bbox": [1, 1, 11, 11], "detection_confidence": .9},
        {"bbox": [30, 30, 40, 40], "detection_confidence": .7},
    ]
    kept = deduplicate(rows, overlap=.6)
    assert [row["detection_confidence"] for row in kept] == [.9, .7]

"""Group saved per-view metrics by held-out versus seen sun directions."""

import argparse
import hashlib
import json
from pathlib import Path


def group_metrics(run):
    config = json.loads((run / "training_config.json").read_text(encoding="utf-8"))
    if not config["model"]["eval"]:
        raise ValueError("Evaluation grouping requires a held-out training split")
    source = Path(config["model"]["source_path"])
    train_json = source / "transforms_train.json"
    test_json = source / "transforms_test.json"
    training = json.loads(train_json.read_text(encoding="utf-8"))["frames"]
    testing = json.loads(test_json.read_text(encoding="utf-8"))["frames"]
    train_suns = {frame["time_index"] for frame in training}
    heldout = {frame["time_index"] for frame in testing} - train_suns
    expected = {frame["file_path"]: frame["time_index"] for frame in testing}
    per_view = json.loads((run / "per_view.json").read_text(encoding="utf-8"))
    results = {}
    for method, metrics in per_view.items():
        manifest = json.loads((run / "test" / method / "manifest.json").read_text(encoding="utf-8"))
        if {entry["file_path"] for entry in manifest} != set(expected):
            raise ValueError("Render manifest does not match the full test split")
        names = [entry["image"] for entry in manifest]
        if len(names) != len(set(names)) or len(manifest) != len(testing):
            raise ValueError("Duplicate or missing rendered frames")
        groups = {"all": names, "heldout_sun": [], "seen_sun_new_combination": []}
        for entry in manifest:
            if entry["time_index"] != expected[entry["file_path"]]:
                raise ValueError("Render sun ID does not match the dataset")
            group = "heldout_sun" if entry["time_index"] in heldout else "seen_sun_new_combination"
            groups[group].append(entry["image"])
        results[method] = {}
        for group, selected in groups.items():
            values = {"n": len(selected)}
            for metric, scores in metrics.items():
                if set(scores) != set(names):
                    raise ValueError(f"Metric {metric} does not cover the full test set")
                values[metric] = sum(scores[name] for name in selected) / len(selected) if selected else None
            results[method][group] = values
    return {"heldout_sun_ids": sorted(heldout),
            "transforms_train_sha256": hashlib.sha256(train_json.read_bytes()).hexdigest(),
            "transforms_test_sha256": hashlib.sha256(test_json.read_bytes()).hexdigest(),
            "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path)
    args = parser.parse_args()
    run = args.model_path.resolve()
    results = group_metrics(run)
    (run / "grouped_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))

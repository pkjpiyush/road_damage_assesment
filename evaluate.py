"""
evaluate.py
Run after training with:  python evaluate.py

Tests the model on the test images twice: once with the model's own answer, and once
after the centre-line direction check. This shows whether the direction check helps
on YOUR photos, so you can decide whether to keep it (USE_DIRECTION_RULE in config.py).
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from collections import Counter

import cv2
import numpy as np
from sklearn.metrics import classification_report

import config
from analyzer import RoadDamageAnalyzer
from prepare_data import read_image


def main():
    if not config.USE_DIRECTION_RULE:
        raise SystemExit("USE_DIRECTION_RULE is False in config.py. Set it to True to compare.")

    analyzer = RoadDamageAnalyzer()
    names = analyzer.class_names
    y_true, y_model, y_final = [], [], []
    statuses = Counter()
    fixed, broke = [], []

    for true_index, class_name in enumerate(names):
        for path in sorted((config.PROCESSED_DIR / "test" / class_name).glob("*.jpg")):
            image = read_image(path)
            if image is None:
                continue
            result = analyzer.analyze(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            model_index = names.index(result["model_class_name"])
            final_index = names.index(result["class_name"])
            y_true.append(true_index)
            y_model.append(model_index)
            y_final.append(final_index)

            if result["direction_check"]:
                statuses[result["direction_check"]["status"]] += 1
            if model_index != true_index and final_index == true_index:
                fixed.append(f"{class_name}/{path.name}")
            if model_index == true_index and final_index != true_index:
                broke.append(f"{class_name}/{path.name}")

    y_true, y_model, y_final = map(np.array, (y_true, y_model, y_final))
    model_acc, final_acc = np.mean(y_true == y_model), np.mean(y_true == y_final)

    lines = [
        "=== Model only ===",
        classification_report(y_true, y_model, target_names=names, digits=3, zero_division=0),
        "=== Model + direction check ===",
        classification_report(y_true, y_final, target_names=names, digits=3, zero_division=0),
        f"Accuracy, model only:             {model_acc:.1%}",
        f"Accuracy, with direction check:   {final_acc:.1%}",
        "",
        "What the direction check did on longitudinal/transverse predictions:",
        f"  confirmed the model:     {statuses['confirmed']}",
        f"  changed the answer:      {statuses['changed']}",
        f"  no centre line found:    {statuses['no_line']}",
        f"  crack direction unclear: {statuses['unclear']}",
        "",
        f"Fixed by the direction check ({len(fixed)}): " + (", ".join(fixed) or "none"),
        f"Broken by the direction check ({len(broke)}): " + (", ".join(broke) or "none"),
        "",
    ]
    if final_acc > model_acc:
        lines.append("Verdict: the direction check HELPED. Keep USE_DIRECTION_RULE = True.")
    elif final_acc < model_acc:
        lines.append("Verdict: the direction check HURT. Look at the 'broken' photos, or set "
                     "USE_DIRECTION_RULE = False in config.py.")
    else:
        lines.append("Verdict: no difference on these test photos.")

    report = "\n".join(lines)
    print(report)
    (config.RESULTS_DIR / "direction_check_report.txt").write_text(report, encoding="utf-8")
    print(f"\nSaved to {config.RESULTS_DIR / 'direction_check_report.txt'}")


if __name__ == "__main__":
    main()
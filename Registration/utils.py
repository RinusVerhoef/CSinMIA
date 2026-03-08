import os, re


def final_metric_from_elastix_log():
    log_path = "elastix.log"
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"No elastix.log found in current dir")

    pat = re.compile(r"Final metric value\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
    metric = None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if m:
                metric = float(m.group(1))
    if metric is None:
        raise RuntimeError("Could not find 'Final metric value' in elastix.log")
    return -metric

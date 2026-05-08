from __future__ import annotations

from pathlib import Path

import pandas as pd


def collect_metric_tables(output_dir: str | Path) -> pd.DataFrame:
    rows = []
    for path in Path(output_dir).glob("tables/*_test_metrics.json"):
        rows.append(pd.read_json(path, typ="series").to_dict() | {"run": path.stem})
    return pd.DataFrame(rows)


def write_summary_table(output_dir: str | Path) -> Path:
    frame = collect_metric_tables(output_dir)
    destination = Path(output_dir) / "tables" / "summary.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination

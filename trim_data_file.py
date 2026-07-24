from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Sequence

STANDARD_HEADER = [
    "ERM Mass",
    "Voltage (V)",
    "Current (mA)",
    "Peak Frequency",
    "Depth Measured (mm)",
    "Total Depth (mm)",
    "Power (mW)",
]


def _find_data_header(rows: Sequence[Sequence[str]]) -> tuple[int, list[str]]:
    """Find the first header row that contains the expected measurement columns."""
    for index, row in enumerate(rows):
        if not row:
            continue

        normalized = [cell.strip().lower() for cell in row if cell.strip()]
        if not normalized:
            continue

        if all(
            field in " ".join(normalized)
            for field in [
                "erm mass",
                "voltage (v)",
                "current (ma)",
                "peak frequency",
                "depth measured (mm)",
                "total depth (mm)",
            ]
        ):
            return index, [cell.strip() for cell in row]

    raise ValueError("Could not find a valid data header row in the CSV file.")


def trim_csv_to_consistent_format(csv_path: str | Path, output_path: str | Path | None = None) -> List[List[str]]:
    """Trim a messy exported CSV into the same consistent format used by frequencyVdepth.csv.

    The function removes Excel-export fluff, identifies the real data header, keeps only the
    standard columns, and optionally writes the cleaned result to a new CSV file.
    """
    input_path = Path(csv_path)
    rows = list(csv.reader(input_path.read_text(encoding="utf-8-sig", errors="ignore").splitlines()))

    header_index, raw_header = _find_data_header(rows)
    header_map = {cell.strip().lower(): idx for idx, cell in enumerate(raw_header)}

    source_columns = {
        "erm mass": header_map.get("erm mass") or header_map.get("erm mass (g?)"),
        "voltage": header_map.get("voltage (v)"),
        "current": header_map.get("current (ma)"),
        "frequency": header_map.get("peak frequency"),
        "depth measured": header_map.get("depth measured (mm)"),
        "total depth": header_map.get("total depth (mm)"),
        "power": header_map.get("power (mw)") or header_map.get("final power (mw)"),
    }

    missing = [name for name, index in source_columns.items() if index is None]
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {', '.join(missing)}")

    cleaned_rows: List[List[str]] = []
    cleaned_rows.append(STANDARD_HEADER)

    for row in rows[header_index + 1 :]:
        if not row or all(not cell.strip() for cell in row):
            continue

        if len(row) <= max(source_columns.values()):
            continue

        cleaned_rows.append(
            [
                row[source_columns["erm mass"]],
                row[source_columns["voltage"]],
                row[source_columns["current"]],
                row[source_columns["frequency"]],
                row[source_columns["depth measured"]],
                row[source_columns["total depth"]],
                row[source_columns["power"]],
            ]
        )

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerows(cleaned_rows)

    return cleaned_rows


if __name__ == "__main__":
    source = Path("Data/frequencyVdepth4.csv")
    destination = Path("Data/frequencyVdepth4_trimmed.csv")
    trimmed = trim_csv_to_consistent_format(source, destination)
    print(f"Trimmed {len(trimmed) - 1} data rows into {destination}")

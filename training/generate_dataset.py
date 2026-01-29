import argparse
import csv
import io
import json
from pathlib import Path
from urllib.request import urlopen

import pandas as pd


OPENML_DATA_ID = 42178
OPENML_API_JSON = f"https://www.openml.org/api/v1/json/data/{OPENML_DATA_ID}"
OPENML_API_JSON_FALLBACK = f"https://old.openml.org/api/v1/json/data/{OPENML_DATA_ID}"


def fetch_openml_dataset_url() -> str:
    for api_url in (OPENML_API_JSON, OPENML_API_JSON_FALLBACK):
        try:
            with urlopen(api_url) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return payload["data_set_description"]["url"]
        except Exception:
            continue
    raise RuntimeError("Failed to resolve OpenML dataset download URL")


def parse_arff(arff_text: str) -> pd.DataFrame:
    lines = arff_text.splitlines()
    columns = []
    data_started = False
    data_rows = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        lower = line.lower()
        if lower.startswith("@attribute"):
            parts = line.split(None, 2)
            if len(parts) >= 2:
                name = parts[1].strip()
                if name.startswith("'") and name.endswith("'"):
                    name = name[1:-1]
                columns.append(name)
        elif lower.startswith("@data"):
            data_started = True
            continue
        elif data_started:
            data_rows.append(line)

    reader = csv.reader(io.StringIO("\n".join(data_rows)), delimiter=",", quotechar="'")
    data = list(reader)
    return pd.DataFrame(data, columns=columns)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Telco Customer Churn dataset from OpenML and save as CSV."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "churn_dataset.csv",
    )
    args = parser.parse_args()

    dataset_url = fetch_openml_dataset_url()
    with urlopen(dataset_url) as response:
        arff_text = response.read().decode("utf-8", errors="replace")

    df = parse_arff(arff_text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved dataset to {args.output} with {len(df)} rows")


if __name__ == "__main__":
    main()

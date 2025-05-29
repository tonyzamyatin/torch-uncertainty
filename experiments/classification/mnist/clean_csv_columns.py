import sys

import pandas as pd


def clean_csv_columns(input_csv, output_csv=None):
    df = pd.read_csv(input_csv)
    df.columns = [col.replace("↑", "").replace("↓", "").strip() for col in df.columns]
    if output_csv is None:
        output_csv = input_csv  # Overwrite original
    df.to_csv(output_csv, index=False)
    print(f"Cleaned CSV saved to {output_csv}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clean_csv_columns.py <input_csv> [output_csv]")
        sys.exit(1)
    input_csv = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None
    clean_csv_columns(input_csv, output_csv)

import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

SOURCE_FILE_COUNT = 3
ROWS_PER_FILE = 100
RANDOM_SEED = 42
FIRST_EVENT_TIME = datetime(2025, 1, 1, tzinfo=UTC)


def generate_input_files(input_directory: Path) -> list[Path]:
    input_directory.mkdir(parents=True, exist_ok=True)
    random_generator = random.Random(RANDOM_SEED)
    input_files = []

    for file_index in range(SOURCE_FILE_COUNT):
        first_event_id = file_index * ROWS_PER_FILE + 1
        rows = []
        for event_id in range(first_event_id, first_event_id + ROWS_PER_FILE):
            rows.append(
                {
                    "event_id": event_id,
                    "event_time": FIRST_EVENT_TIME + timedelta(seconds=event_id - 1),
                    "page_path": random_generator.choice(("/", "/catalog", "/help")),
                    "event_type": random_generator.choice(("page_view", "click")),
                    "session_id": random_generator.randint(1, 50),
                }
            )

        input_file = input_directory / f"events-{file_index + 1:03d}.parquet"
        pd.DataFrame(rows).to_parquet(
            input_file, engine="pyarrow", compression="snappy", index=False
        )
        input_files.append(input_file)

    return input_files


def compact_with_pandas(input_files: list[Path], output_file: Path) -> int:
    dataframes = [pd.read_parquet(path, engine="pyarrow") for path in input_files]
    compacted = pd.concat(dataframes, ignore_index=True)
    compacted.to_parquet(
        output_file, engine="pyarrow", compression="snappy", index=False
    )
    return len(compacted)


def main() -> None:
    data_directory = Path(__file__).resolve().parent / "benchmark_data"
    input_files = generate_input_files(data_directory / "inputs")
    output_file = data_directory / "pandas.parquet"
    row_count = compact_with_pandas(input_files, output_file)
    print(f"Compacted {len(input_files)} files into {output_file} ({row_count} rows).")


if __name__ == "__main__":
    main()

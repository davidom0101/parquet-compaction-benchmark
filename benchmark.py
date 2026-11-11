import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd

SOURCE_FILE_COUNT = 3
ROWS_PER_FILE = 100

# A fixed seed and start time make the input rows reproducible.
RANDOM_SEED = 42
FIRST_EVENT_TIME = datetime(2025, 1, 1, tzinfo=UTC)


def generate_input_files(input_directory: Path) -> list[Path]:
    input_directory.mkdir(parents=True, exist_ok=True)
    random_generator = random.Random(RANDOM_SEED)
    input_files = []

    for file_index in range(SOURCE_FILE_COUNT):
        # Give each file its own range of event IDs.
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
    # This baseline holds the source DataFrames and the combined result in memory.
    dataframes = [pd.read_parquet(path, engine="pyarrow") for path in input_files]
    compacted = pd.concat(dataframes, ignore_index=True)
    compacted.to_parquet(
        output_file, engine="pyarrow", compression="snappy", index=False
    )
    return len(compacted)


def compact_with_duckdb(input_files: list[Path], output_file: Path) -> None:
    # DuckDB can stream rows from the Parquet scan directly to the output writer.
    with duckdb.connect() as connection:
        connection.execute(
            """
            COPY (SELECT * FROM read_parquet($input_files))
            TO $output_file (FORMAT PARQUET, COMPRESSION SNAPPY)
            """,
            {
                "input_files": [str(path) for path in input_files],
                "output_file": str(output_file),
            },
        )


def validate_outputs(input_files: list[Path], output_files: list[Path]) -> None:
    source_rows = pd.concat(
        [pd.read_parquet(path, engine="pyarrow") for path in input_files],
        ignore_index=True,
    ).sort_values("event_id", ignore_index=True)

    for output_file in output_files:
        try:
            output_rows = pd.read_parquet(output_file, engine="pyarrow").sort_values(
                "event_id", ignore_index=True
            )
            # The writers can represent the same values with different dtypes.
            pd.testing.assert_frame_equal(
                source_rows, output_rows, check_dtype=False, check_exact=True
            )
        except (AssertionError, KeyError) as error:
            raise ValueError(
                f"{output_file.name} does not match the source rows"
            ) from error


def main() -> None:
    data_directory = Path(__file__).resolve().parent / "benchmark_data"
    input_files = generate_input_files(data_directory / "inputs")
    pandas_output = data_directory / "pandas.parquet"
    row_count = compact_with_pandas(input_files, pandas_output)
    print(
        f"Compacted {len(input_files)} files into {pandas_output} ({row_count} rows)."
    )

    duckdb_output = data_directory / "duckdb.parquet"
    compact_with_duckdb(input_files, duckdb_output)
    print(f"Compacted {len(input_files)} files into {duckdb_output}.")

    validate_outputs(input_files, [pandas_output, duckdb_output])
    print("Both outputs match the source rows.")


if __name__ == "__main__":
    main()

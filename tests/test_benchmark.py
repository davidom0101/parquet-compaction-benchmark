from pathlib import Path

import pandas as pd
import pytest

from benchmark import validate_outputs


def test_validation_accepts_reordered_rows(tmp_path: Path) -> None:
    source_file = tmp_path / "source.parquet"
    output_file = tmp_path / "output.parquet"
    rows = pd.DataFrame({"event_id": [1, 2], "page_path": ["/", "/help"]})
    rows.to_parquet(source_file, index=False)
    rows.iloc[::-1].to_parquet(output_file, index=False)

    validate_outputs([source_file], [output_file])


@pytest.mark.parametrize(
    ("event_ids", "page_paths"),
    [
        ([1], ["/"]),
        ([1, 2], ["/", "/changed"]),
        ([1, 1], ["/", "/help"]),
    ],
    ids=["missing-row", "changed-value", "duplicate-id"],
)
def test_validation_rejects_different_rows(
    tmp_path: Path, event_ids: list[int], page_paths: list[str]
) -> None:
    source_file = tmp_path / "source.parquet"
    output_file = tmp_path / "output.parquet"
    pd.DataFrame({"event_id": [1, 2], "page_path": ["/", "/help"]}).to_parquet(
        source_file, index=False
    )
    pd.DataFrame({"event_id": event_ids, "page_path": page_paths}).to_parquet(
        output_file, index=False
    )

    with pytest.raises(ValueError, match=r"output\.parquet does not match"):
        validate_outputs([source_file], [output_file])

# tests/test_saver.py
from pathlib import Path
import pyarrow.parquet as pq

from intake.saver import save_record_to_parquet

def minimal_record():
    return {
        "proposal_type": "new_collection",
        "title": "T",
        "pi_name": "P",
        "contact_email": "p@x.com",
        "org_name": "O",
        "data_modalities": ["CT", "MR"],
        "data_modalities_other": "",
        "short_abstract": "A",
    }

def test_save_record_creates_partitions(tmp_path):
    root = tmp_path / "parquet"
    rec = minimal_record()
    save_record_to_parquet(rec, root=str(root))

    # Confirm a Parquet dataset exists under partitioned dirs
    parts = list(root.rglob("*.parquet"))
    assert parts, "Expected Parquet files to be written"

    # Read dataset and verify columns
    ds = pq.ParquetDataset(str(root))
    table = ds.read()
    cols = set(table.column_names)
    expected = {
        "id", "created_at", "proposal_type", "title", "pi_name",
        "contact_email", "org_name", "data_modalities",
        "data_modalities_other", "short_abstract"
    }
    assert expected.issubset(cols)

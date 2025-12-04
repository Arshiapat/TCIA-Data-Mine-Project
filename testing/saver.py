# intake/saver.py
from __future__ import annotations
from pathlib import Path
from uuid import uuid4
from datetime import datetime, timezone
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def save_record_to_parquet(rec: dict, root: str | Path = "./data/parquet") -> Path:
    """
    Append a single record into a partitioned Parquet dataset:
      root/proposal_type=<type>/dt=YYYY-MM-DD/part-*.parquet
    Returns the dataset root path.
    """
    rec = {**rec}
    # timezone-aware UTC timestamp (avoids deprecation warning)
    rec["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rec["id"] = str(uuid4())

    # partition value
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rec["dt"] = today  # <-- MUST exist in the table if we partition on it

    base = Path(root)
    (base / f"proposal_type={rec['proposal_type']}" / f"dt={today}").mkdir(parents=True, exist_ok=True)

    cols = [
        "id", "created_at", "proposal_type", "dt",  # include dt in schema
        "title", "pi_name", "contact_email", "org_name",
        "data_modalities", "data_modalities_other", "short_abstract",
    ]
    df = pd.DataFrame([{k: rec.get(k, None) for k in cols}])
    df["data_modalities"] = df["data_modalities"].apply(lambda x: x if isinstance(x, list) else [])

    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_to_dataset(
        table,
        root_path=str(base),
        partition_cols=["proposal_type", "dt"],  # now both exist in the table
    )
    return base

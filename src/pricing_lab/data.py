"""Download, convert and query the UCI Online Retail II dataset with DuckDB."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import duckdb
import pandas as pd
import requests

UCI_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
SQL_DIR = ROOT / "sql"
RAW_PARQUET = DATA_DIR / "online_retail_ii.parquet"


def download_raw(force: bool = False) -> Path:
    """Fetch the UCI archive and convert both Excel sheets into a single Parquet file."""
    if RAW_PARQUET.exists() and not force:
        return RAW_PARQUET
    DATA_DIR.mkdir(exist_ok=True)
    zip_path = DATA_DIR / "online_retail_ii.zip"
    if not zip_path.exists():
        resp = requests.get(UCI_URL, timeout=300)
        resp.raise_for_status()
        zip_path.write_bytes(resp.content)
    with zipfile.ZipFile(zip_path) as zf:
        xlsx_name = next(n for n in zf.namelist() if n.endswith(".xlsx"))
        sheets = pd.read_excel(
            io.BytesIO(zf.read(xlsx_name)), sheet_name=None,
            dtype={"Invoice": str, "StockCode": str, "Description": str, "Country": str},
        )
    df = pd.concat(sheets.values(), ignore_index=True)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    df.to_parquet(RAW_PARQUET, index=False)
    return RAW_PARQUET


def connect() -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB session and build the analytical views defined in sql/."""
    con = duckdb.connect()
    con.execute(f"CREATE VIEW raw AS SELECT * FROM read_parquet('{RAW_PARQUET.as_posix()}')")
    for sql_file in sorted(SQL_DIR.glob("*.sql")):
        con.execute(sql_file.read_text(encoding="utf-8"))
    return con

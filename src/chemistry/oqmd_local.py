import csv
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .oqmd import normalize_optimade_formula
from .schema import ChemicalProperty, ChemicalRecord


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = PROJECT_ROOT / "data" / "sources" / "oqmd_cgnn"
CSV_PATH = SOURCE_DIR / "targets.csv"
LICENSE_PATH = SOURCE_DIR / "License"
INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "oqmd.sqlite"
INDEX_SCHEMA_VERSION = 2

NUMERIC_PROPERTIES = {
    "energy_per_atom": ("energy_per_atom", "eV/atom"),
    "formation_energy_per_atom": (
        "formation_energy_per_atom",
        "eV/atom",
    ),
    "band_gap": ("band_gap", "eV"),
    "volume_per_atom": ("volume_per_atom", "Å³/atom"),
    "magnetization_per_atom": (
        "magnetization_per_atom",
        "μB/atom",
    ),
    "atomic_volume_per_atom": (
        "atomic_volume_per_atom",
        "Å³/atom",
    ),
    "volume_deviation": ("volume_deviation", None),
}


def local_oqmd_available() -> bool:
    return CSV_PATH.is_file()


def _index_is_current() -> bool:
    if not INDEX_PATH.is_file() or not CSV_PATH.is_file():
        return False

    if INDEX_PATH.stat().st_mtime_ns < CSV_PATH.stat().st_mtime_ns:
        return False

    try:
        connection = sqlite3.connect(INDEX_PATH)
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        connection.close()
    except sqlite3.Error:
        return False

    return version == INDEX_SCHEMA_VERSION


def build_local_oqmd_index(force: bool = False) -> Path:
    if not CSV_PATH.is_file():
        raise FileNotFoundError(
            f"Local OQMD targets were not found at {CSV_PATH}."
        )

    if not force and _index_is_current():
        return INDEX_PATH

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = INDEX_PATH.with_suffix(".sqlite.tmp")
    temporary_path.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary_path)

    try:
        connection.executescript(
            """
            PRAGMA journal_mode = OFF;
            PRAGMA synchronous = OFF;
            PRAGMA temp_store = MEMORY;
            CREATE TABLE materials (
                name TEXT PRIMARY KEY,
                formula TEXT NOT NULL,
                formula_key TEXT NOT NULL,
                spacegroup INTEGER,
                nelements INTEGER,
                nsites INTEGER,
                energy_per_atom REAL,
                formation_energy_per_atom REAL,
                band_gap REAL,
                volume_per_atom REAL,
                magnetization_per_atom REAL,
                atomic_volume_per_atom REAL,
                volume_deviation REAL
            );
            """
        )
        source_columns = (
            "name", "formula", "spacegroup", "nelements", "nsites",
            "energy_per_atom", "formation_energy_per_atom", "band_gap",
            "volume_per_atom", "magnetization_per_atom",
            "atomic_volume_per_atom", "volume_deviation",
        )
        insert_columns = (
            "name", "formula", "formula_key", "spacegroup", "nelements",
            "nsites", "energy_per_atom", "formation_energy_per_atom",
            "band_gap", "volume_per_atom", "magnetization_per_atom",
            "atomic_volume_per_atom", "volume_deviation",
        )
        placeholders = ",".join("?" for _ in insert_columns)
        insert_sql = (
            f"INSERT INTO materials ({','.join(insert_columns)}) "
            f"VALUES ({placeholders})"
        )

        with CSV_PATH.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            batch = []

            for row in reader:
                formula = row["formula"]

                try:
                    formula_key = normalize_optimade_formula(formula)
                except ValueError:
                    formula_key = formula

                values = [row[column] or None for column in source_columns]
                values.insert(2, formula_key)
                batch.append(tuple(values))

                if len(batch) == 10_000:
                    connection.executemany(insert_sql, batch)
                    batch.clear()

            if batch:
                connection.executemany(insert_sql, batch)

        connection.executescript(
            """
            CREATE INDEX materials_formula_idx ON materials(formula_key);
            CREATE INDEX materials_formation_idx
                ON materials(formula_key, formation_energy_per_atom);
            """
        )
        connection.execute(f"PRAGMA user_version = {INDEX_SCHEMA_VERSION}")
        connection.commit()
    finally:
        connection.close()

    temporary_path.replace(INDEX_PATH)
    return INDEX_PATH


def _property(name: str, value: object, unit: str | None) -> ChemicalProperty:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise TypeError(f"Invalid numeric OQMD value for {name}: {value!r}")

    return ChemicalProperty(
        name=name,
        value=float(value),
        unit=unit,
        method="OQMD DFT (local CGNN dataset)",
    )


def _record_from_row(row: sqlite3.Row) -> ChemicalRecord:
    properties = [
        _property(property_name, row[column], unit)
        for column, (property_name, unit) in NUMERIC_PROPERTIES.items()
        if row[column] is not None
    ]
    oqmd_id = str(row["name"]).removeprefix("oqmd-")

    return ChemicalRecord(
        record_id=f"oqmd-local:{oqmd_id}",
        domain="inorganic",
        entity_kind="crystal",
        source="OQMD local",
        source_id=oqmd_id,
        preferred_name=str(row["formula"]),
        formula=str(row["formula"]),
        representations={"oqmd_id": str(row["name"])},
        properties=properties,
        source_url=f"https://oqmd.org/materials/entry/{oqmd_id}",
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        license_name="CC-BY-4.0",
        metadata={
            "space_group_number": row["spacegroup"],
            "number_of_elements": row["nelements"],
            "number_of_sites": row["nsites"],
            "dataset": "OQMD v1.2 for CGNN",
        },
    )


def _connect() -> sqlite3.Connection:
    build_local_oqmd_index()
    connection = sqlite3.connect(INDEX_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def search_local_oqmd_by_formula(
    formula: str,
    limit: int = 5,
) -> list[ChemicalRecord]:
    if limit <= 0:
        raise ValueError("The limit must be greater than zero.")

    connection = _connect()

    try:
        rows = connection.execute(
            """
            SELECT * FROM materials
            WHERE formula_key = ?
            ORDER BY formation_energy_per_atom ASC
            LIMIT ?
            """,
            (normalize_optimade_formula(formula), limit),
        ).fetchall()
    finally:
        connection.close()

    return [_record_from_row(row) for row in rows]


def fetch_local_oqmd_page(
    offset: int,
    limit: int = 100,
) -> list[ChemicalRecord]:
    if offset < 0:
        raise ValueError("The offset cannot be negative.")

    if limit <= 0:
        raise ValueError("The limit must be greater than zero.")

    connection = _connect()

    try:
        rows = connection.execute(
            "SELECT * FROM materials ORDER BY name LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    finally:
        connection.close()

    return [_record_from_row(row) for row in rows]

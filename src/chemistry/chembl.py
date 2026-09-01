from datetime import datetime, timezone

from .http_client import ChemistryApiError, get_json
from .schema import ChemicalProperty, ChemicalRecord


CHEMBL_SEARCH_URL = (
    "https://www.ebi.ac.uk/chembl/api/data/molecule/search.json"
)
CHEMBL_MOLECULE_URL = (
    "https://www.ebi.ac.uk/chembl/api/data/molecule.json"
)


def _coerce_number(value: object) -> str | int | float | bool:
    if isinstance(value, (int, float, bool)):
        return value

    text = str(value)

    try:
        number = float(text)
    except ValueError:
        return text

    return int(number) if number.is_integer() else number


def _add_property(
    properties: list[ChemicalProperty],
    source: dict,
    source_key: str,
    name: str,
    unit: str | None = None,
) -> None:
    value = source.get(source_key)

    if value is not None:
        properties.append(
            ChemicalProperty(
                name=name,
                value=_coerce_number(value),
                unit=unit,
                method="ChEMBL calculated property",
            )
        )


def _record_from_molecule(row: dict) -> ChemicalRecord | None:
    chembl_id = row.get("molecule_chembl_id")

    if not chembl_id:
        return None

    structures = row.get("molecule_structures") or {}
    molecule_properties = row.get("molecule_properties") or {}
    representations = {}

    representation_fields = {
        "canonical_smiles": structures.get("canonical_smiles"),
        "inchi": structures.get("standard_inchi"),
        "inchikey": structures.get("standard_inchi_key"),
    }

    for name, value in representation_fields.items():
        if value:
            representations[name] = str(value)

    properties: list[ChemicalProperty] = []
    _add_property(
        properties,
        molecule_properties,
        "full_mwt",
        "molecular_weight",
        "g/mol",
    )
    _add_property(
        properties,
        molecule_properties,
        "alogp",
        "alogp",
    )
    _add_property(
        properties,
        molecule_properties,
        "psa",
        "polar_surface_area",
        "Å²",
    )
    _add_property(
        properties,
        molecule_properties,
        "hba",
        "hydrogen_bond_acceptor_count",
    )
    _add_property(
        properties,
        molecule_properties,
        "hbd",
        "hydrogen_bond_donor_count",
    )
    _add_property(
        properties,
        molecule_properties,
        "rtb",
        "rotatable_bond_count",
    )

    preferred_name = row.get("pref_name") or str(chembl_id)

    return ChemicalRecord(
        record_id=f"chembl:{chembl_id}",
        domain="organic",
        entity_kind="molecule",
        source="ChEMBL",
        source_id=str(chembl_id),
        preferred_name=str(preferred_name),
        formula=molecule_properties.get("full_molformula"),
        representations=representations,
        properties=properties,
        source_url=(
            "https://www.ebi.ac.uk/chembl/explore/compound/"
            f"{chembl_id}"
        ),
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        license_name="CC-BY-SA-3.0",
        metadata={
            "molecule_type": row.get("molecule_type"),
            "query_source": "ChEMBL full-text search",
        },
    )


def search_chembl_molecules(
    query: str,
    limit: int = 3,
) -> list[ChemicalRecord]:
    if limit <= 0:
        raise ValueError("The limit must be greater than zero.")

    payload = get_json(
        CHEMBL_SEARCH_URL,
        params={"q": query, "limit": limit},
        timeout_seconds=20,
        max_attempts=2,
    )
    records = []

    for row in payload.get("molecules", []):
        record = _record_from_molecule(row)

        if record is not None:
            records.append(record)

    if not records:
        raise ChemistryApiError(
            f"ChEMBL did not find a molecule for: {query}"
        )

    return records


def fetch_chembl_molecule_page(
    offset: int,
    limit: int = 100,
) -> list[ChemicalRecord]:
    if offset < 0:
        raise ValueError("The offset cannot be negative.")

    if limit <= 0:
        raise ValueError("The limit must be greater than zero.")

    payload = get_json(
        CHEMBL_MOLECULE_URL,
        params={
            "limit": limit,
            "offset": offset,
            "molecule_structures__isnull": "false",
            "molecule_properties__isnull": "false",
        },
        timeout_seconds=30,
        max_attempts=3,
    )
    records = []

    for row in payload.get("molecules", []):
        record = _record_from_molecule(row)

        if record is not None and record.representations.get(
            "canonical_smiles"
        ):
            records.append(record)

    return records

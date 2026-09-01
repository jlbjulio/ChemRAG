from datetime import datetime, timezone
from urllib.parse import quote

from .http_client import ChemistryApiError, get_json
from .schema import ChemicalProperty, ChemicalRecord


PUBCHEM_BASE_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
)
PUBCHEM_PROPERTIES = ",".join(
    [
        "Title",
        "IUPACName",
        "CanonicalSMILES",
        "IsomericSMILES",
        "InChI",
        "InChIKey",
        "MolecularFormula",
        "MolecularWeight",
        "XLogP",
        "TPSA",
        "HBondDonorCount",
        "HBondAcceptorCount",
        "RotatableBondCount",
    ]
)
ALLOWED_NAMESPACES = {"name", "cid", "smiles", "inchikey"}


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
                value=value,
                unit=unit,
                method="PubChem record",
            )
        )


def fetch_pubchem_compound(
    identifier: str,
    namespace: str = "name",
) -> ChemicalRecord:
    if namespace not in ALLOWED_NAMESPACES:
        allowed = ", ".join(sorted(ALLOWED_NAMESPACES))
        raise ValueError(
            f"Unsupported namespace: {namespace}. Use one of: {allowed}."
        )

    encoded_identifier = quote(identifier.strip(), safe="")
    url = (
        f"{PUBCHEM_BASE_URL}/{namespace}/{encoded_identifier}/property/"
        f"{PUBCHEM_PROPERTIES}/JSON"
    )
    payload = get_json(url)
    rows = payload.get("PropertyTable", {}).get("Properties", [])

    if not rows:
        raise ChemistryApiError(
            f"PubChem did not find the compound: {identifier}"
        )

    row = rows[0]
    cid = str(row["CID"])
    representations = {}

    representation_fields = {
        "canonical_smiles": (
            row.get("ConnectivitySMILES")
            or row.get("CanonicalSMILES")
        ),
        "isomeric_smiles": (
            row.get("SMILES")
            or row.get("IsomericSMILES")
        ),
        "inchi": row.get("InChI"),
        "inchikey": row.get("InChIKey"),
    }

    for name, value in representation_fields.items():
        if value:
            representations[name] = str(value)

    properties: list[ChemicalProperty] = []
    _add_property(
        properties,
        row,
        "MolecularWeight",
        "molecular_weight",
        "g/mol",
    )
    _add_property(properties, row, "XLogP", "xlogp")
    _add_property(properties, row, "TPSA", "tpsa", "Å²")
    _add_property(
        properties,
        row,
        "HBondDonorCount",
        "hydrogen_bond_donor_count",
    )
    _add_property(
        properties,
        row,
        "HBondAcceptorCount",
        "hydrogen_bond_acceptor_count",
    )
    _add_property(
        properties,
        row,
        "RotatableBondCount",
        "rotatable_bond_count",
    )

    return ChemicalRecord(
        record_id=f"pubchem:{cid}",
        domain="organic",
        entity_kind="molecule",
        source="PubChem",
        source_id=cid,
        preferred_name=str(
            row.get("Title")
            or row.get("IUPACName")
            or identifier
        ),
        formula=row.get("MolecularFormula"),
        representations=representations,
        properties=properties,
        source_url=f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        metadata={
            "iupac_name": row.get("IUPACName"),
            "query": identifier,
            "query_namespace": namespace,
        },
    )

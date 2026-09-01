import re
from datetime import datetime, timezone

from .http_client import ChemistryApiError, get_json
from .schema import ChemicalProperty, ChemicalRecord


OQMD_OPTIMADE_URL = "https://oqmd.org/optimade/structures"
OQMD_RESPONSE_FIELDS = ",".join(
    [
        "id",
        "chemical_formula_reduced",
        "chemical_formula_descriptive",
        "elements",
        "nsites",
        "space_group_symbol_hermann_mauguin",
        "space_group_it_number",
        "_oqmd_entry_id",
        "_oqmd_band_gap",
        "_oqmd_delta_e",
        "_oqmd_stability",
        "_oqmd_volume",
        "_oqmd_prototype",
    ]
)
FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*(?:\.\d+)?)")


def normalize_optimade_formula(formula: str) -> str:
    compact_formula = formula.replace(" ", "")
    matches = list(FORMULA_TOKEN.finditer(compact_formula))

    if not matches or "".join(match.group(0) for match in matches) != compact_formula:
        raise ValueError(
            "Use a simple formula such as BaTiO3 or SiO2."
        )

    amounts: dict[str, float] = {}

    for match in matches:
        element, amount_text = match.groups()
        amount = float(amount_text) if amount_text else 1.0
        amounts[element] = amounts.get(element, 0.0) + amount

    parts = []

    for element in sorted(amounts):
        amount = amounts[element]

        if amount == 1:
            amount_text = ""
        elif amount.is_integer():
            amount_text = str(int(amount))
        else:
            amount_text = str(amount)

        parts.append(f"{element}{amount_text}")

    return "".join(parts)


def _add_property(
    properties: list[ChemicalProperty],
    attributes: dict,
    source_key: str,
    name: str,
    unit: str | None,
) -> None:
    value = attributes.get(source_key)

    if value is not None:
        properties.append(
            ChemicalProperty(
                name=name,
                value=value,
                unit=unit,
                method="OQMD DFT",
            )
        )


def _record_from_row(
    row: dict,
    query_formula: str | None = None,
) -> ChemicalRecord:
    attributes = row.get("attributes", {})
    optimade_id = str(row["id"])
    entry_id = str(attributes.get("_oqmd_entry_id") or optimade_id)
    properties: list[ChemicalProperty] = []
    _add_property(
        properties,
        attributes,
        "_oqmd_band_gap",
        "band_gap",
        "eV",
    )
    _add_property(
        properties,
        attributes,
        "_oqmd_delta_e",
        "formation_energy_per_atom",
        "eV/atom",
    )
    _add_property(
        properties,
        attributes,
        "_oqmd_stability",
        "distance_from_convex_hull",
        "eV/atom",
    )
    _add_property(
        properties,
        attributes,
        "_oqmd_volume",
        "unit_cell_volume",
        "Å³",
    )
    formula = str(
        attributes.get("chemical_formula_descriptive")
        or attributes.get("chemical_formula_reduced")
        or query_formula
        or optimade_id
    )
    representations = {
        "optimade_id": optimade_id,
        "optimade_formula": str(
            attributes.get("chemical_formula_reduced") or formula
        ),
    }

    if query_formula:
        representations["query_formula"] = query_formula

    return ChemicalRecord(
        record_id=f"oqmd:{optimade_id}",
        domain="inorganic",
        entity_kind="crystal",
        source="OQMD",
        source_id=entry_id,
        preferred_name=query_formula or formula,
        formula=formula,
        representations=representations,
        properties=properties,
        source_url=f"https://oqmd.org/materials/entry/{entry_id}",
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        license_name="CC-BY-4.0",
        metadata={
            "query": query_formula,
            "elements": attributes.get("elements", []),
            "number_of_sites": attributes.get("nsites"),
            "space_group": attributes.get(
                "space_group_symbol_hermann_mauguin"
            ),
            "space_group_number": attributes.get(
                "space_group_it_number"
            ),
            "prototype": attributes.get("_oqmd_prototype"),
        },
    )


def fetch_oqmd_materials(
    formula: str,
    limit: int = 3,
) -> list[ChemicalRecord]:
    if limit <= 0:
        raise ValueError("The limit must be greater than zero.")

    normalized_formula = normalize_optimade_formula(formula)
    payload = get_json(
        OQMD_OPTIMADE_URL,
        params={
            "filter": (
                "chemical_formula_reduced="
                f'"{normalized_formula}"'
            ),
            "page_limit": limit,
            "response_fields": OQMD_RESPONSE_FIELDS,
        },
        timeout_seconds=6,
        max_attempts=1,
    )
    rows = payload.get("data", [])

    if not rows:
        raise ChemistryApiError(
            f"OQMD did not find the material: {formula}"
        )

    return [_record_from_row(row, formula) for row in rows]


def fetch_oqmd_material_page(
    offset: int,
    limit: int = 100,
) -> list[ChemicalRecord]:
    if offset < 0:
        raise ValueError("The offset cannot be negative.")

    if limit <= 0:
        raise ValueError("The limit must be greater than zero.")

    payload = get_json(
        OQMD_OPTIMADE_URL,
        params={
            "page_limit": limit,
            "page_offset": offset,
            "response_fields": OQMD_RESPONSE_FIELDS,
        },
        timeout_seconds=10,
        max_attempts=1,
    )
    return [_record_from_row(row) for row in payload.get("data", [])]

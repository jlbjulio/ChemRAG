import re
from datetime import datetime, timezone

from .http_client import ChemistryApiError, get_json
from .schema import ChemicalProperty, ChemicalRecord


COD_SEARCH_URL = "https://www.crystallography.net/cod/result"
FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*(?:\.\d+)?)")


def _cod_formula(formula: str) -> str:
    compact = formula.replace(" ", "")
    matches = list(FORMULA_TOKEN.finditer(compact))

    if not matches or "".join(match.group(0) for match in matches) != compact:
        raise ValueError("Use a simple formula such as BaTiO3 or SiO2.")

    amounts = {}

    for match in matches:
        element, amount = match.groups()
        amounts[element] = amounts.get(element, 0.0) + (
            float(amount) if amount else 1.0
        )

    if "C" in amounts:
        order = ["C"]

        if "H" in amounts:
            order.append("H")

        order.extend(sorted(set(amounts) - set(order)))
    else:
        order = sorted(amounts)

    parts = []

    for element in order:
        amount = amounts[element]

        if amount == 1:
            amount_text = ""
        elif amount.is_integer():
            amount_text = str(int(amount))
        else:
            amount_text = str(amount)

        parts.append(f"{element}{amount_text}")

    return " ".join(parts)


def _add_numeric_property(
    properties: list[ChemicalProperty],
    row: dict,
    source_key: str,
    name: str,
    unit: str,
) -> None:
    value = row.get(source_key)

    if value in {None, ""}:
        return

    text_value = str(value)

    try:
        numeric_value: str | float = float(text_value)
    except ValueError:
        numeric_value = text_value

    properties.append(
        ChemicalProperty(
            name=name,
            value=numeric_value,
            unit=unit,
            method="COD crystal structure record",
        )
    )


def search_cod_by_formula(
    formula: str,
    limit: int = 5,
) -> list[ChemicalRecord]:
    if limit <= 0:
        raise ValueError("The limit must be greater than zero.")

    payload = get_json(
        COD_SEARCH_URL,
        params={"formula": _cod_formula(formula), "format": "json"},
        timeout_seconds=20,
        max_attempts=2,
    )

    if not isinstance(payload, list) or not payload:
        raise ChemistryApiError(
            f"COD did not find a crystal for: {formula}"
        )

    records = []
    retrieved_at = datetime.now(timezone.utc).isoformat()

    for row in payload[:limit]:
        cod_id = str(row["file"])
        properties: list[ChemicalProperty] = []

        for source_key, name, unit in [
            ("a", "lattice_parameter_a", "Å"),
            ("b", "lattice_parameter_b", "Å"),
            ("c", "lattice_parameter_c", "Å"),
            ("alpha", "lattice_angle_alpha", "degrees"),
            ("beta", "lattice_angle_beta", "degrees"),
            ("gamma", "lattice_angle_gamma", "degrees"),
            ("vol", "unit_cell_volume", "Å³"),
        ]:
            _add_numeric_property(
                properties,
                row,
                source_key,
                name,
                unit,
            )

        preferred_name = (
            row.get("commonname")
            or row.get("chemname")
            or row.get("mineral")
            or formula
        )
        records.append(
            ChemicalRecord(
                record_id=f"cod:{cod_id}",
                domain="inorganic",
                entity_kind="crystal",
                source="COD",
                source_id=cod_id,
                preferred_name=str(preferred_name),
                formula=formula,
                representations={
                    "cod_id": cod_id,
                    "cif_url": (
                        "https://www.crystallography.net/cod/"
                        f"{cod_id}.cif"
                    ),
                },
                properties=properties,
                source_url=(
                    "https://www.crystallography.net/cod/"
                    f"{cod_id}.html"
                ),
                retrieved_at=retrieved_at,
                license_name="CC0-1.0",
                metadata={
                    "space_group": row.get("sg"),
                    "space_group_number": row.get("sgNumber"),
                    "title": row.get("title"),
                    "authors": row.get("authors"),
                    "journal": row.get("journal"),
                    "publication_year": row.get("year"),
                    "doi": row.get("doi"),
                },
            )
        )

    return records

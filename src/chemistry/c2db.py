import html
import re
from datetime import datetime, timezone

from .http_client import ChemistryApiError, get_text
from .schema import ChemicalProperty, ChemicalRecord


C2DB_URL = "https://c2db.fysik.dtu.dk/"
ROW_PATTERN = re.compile(
    r'<tr[^>]*>\s*(.*?)\s*</tr>',
    re.IGNORECASE | re.DOTALL,
)
LINK_PATTERN = re.compile(
    r'<a\s+href=/?material/([^\s>]+)[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
TAG_PATTERN = re.compile(r"<[^>]+>")


def _plain_text(value: str) -> str:
    value = re.sub(r"<sub>(.*?)</sub>", r"\1", value, flags=re.I | re.S)
    return " ".join(
        html.unescape(TAG_PATTERN.sub(" ", value)).split()
    )


def _number(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def search_c2db_by_formula(
    formula: str,
    limit: int = 5,
) -> list[ChemicalRecord]:
    if limit <= 0:
        raise ValueError("The limit must be greater than zero.")

    normalized_formula = formula.replace(" ", "")
    page = get_text(
        C2DB_URL,
        params={"filter": normalized_formula},
        timeout_seconds=20,
        max_attempts=2,
    )
    records = []
    retrieved_at = datetime.now(timezone.utc).isoformat()

    for row_html in ROW_PATTERN.findall(page):
        links = LINK_PATTERN.findall(row_html)

        if len(links) < 6:
            continue

        material_id = links[0][0]
        values = [_plain_text(item[1]) for item in links[:6]]

        if values[0] != normalized_formula:
            continue

        properties = []

        for value, name, unit, method in [
            (values[1], "distance_from_convex_hull", "eV/atom", "C2DB"),
            (values[2], "formation_energy_per_atom", "eV/atom", "C2DB"),
            (values[3], "band_gap", "eV", "C2DB PBE"),
        ]:
            numeric_value = _number(value)

            if numeric_value is not None:
                properties.append(
                    ChemicalProperty(name, numeric_value, unit, method)
                )

        records.append(
            ChemicalRecord(
                record_id=f"c2db:{material_id}",
                domain="inorganic",
                entity_kind="crystal",
                source="C2DB",
                source_id=material_id,
                preferred_name=normalized_formula,
                formula=normalized_formula,
                representations={},
                properties=properties,
                source_url=f"{C2DB_URL}material/{material_id}",
                retrieved_at=retrieved_at,
                license_name="CC-BY-4.0",
                metadata={
                    "magnetic": values[4],
                    "layer_group": values[5],
                    "dimensionality": "2D",
                },
            )
        )

        if len(records) == limit:
            break

    if not records:
        raise ChemistryApiError(
            f"C2DB did not find an exact formula match for: {formula}"
        )

    return records

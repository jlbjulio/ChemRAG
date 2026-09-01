import hashlib
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from gradio_client import Client

from .http_client import ChemistryApiError
from .schema import ChemicalProperty, ChemicalRecord


SLICES_SPACE = "xiaohang07/SLICES"
SLICES_URL = "https://huggingface.co/spaces/xiaohang07/SLICES"


@lru_cache(maxsize=1)
def _client() -> Client:
    return Client(SLICES_SPACE, verbose=False)


def _cif_value(text: str, field: str) -> str | None:
    match = re.search(
        rf"^{re.escape(field)}\s+['\"]?(.+?)['\"]?\s*$",
        text,
        re.MULTILINE,
    )
    return match.group(1).strip(" '\"") if match else None


@lru_cache(maxsize=32)
def decode_slices(slices_string: str) -> ChemicalRecord:
    value = " ".join(slices_string.split())

    if not value:
        raise ValueError("The SLICES string cannot be empty.")

    try:
        cif_path, _, status = _client().predict(
            value,
            api_name="/slices_to_cif",
        )
    except Exception as error:
        raise ChemistryApiError(
            f"The official SLICES converter failed: {error}"
        ) from error

    if not cif_path or not str(status).startswith("Conversion successful"):
        raise ChemistryApiError(
            f"The SLICES string could not be decoded: {status}"
        )

    cif_text = Path(str(cif_path)).read_text(
        encoding="utf-8",
        errors="replace",
    )
    formula = _cif_value(cif_text, "_chemical_formula_structural")

    if not formula:
        raise ChemistryApiError(
            "The decoded CIF did not contain a structural formula."
        )

    properties = []
    energy_match = re.search(r"Energy:\s*([-+0-9.]+)\s*eV/atom", str(status))

    if energy_match:
        properties.append(
            ChemicalProperty(
                "predicted_energy_per_atom",
                float(energy_match.group(1)),
                "eV/atom",
                "SLICES official converter (CHGNet relaxation)",
            )
        )

    for field, name, unit in [
        ("_cell_length_a", "lattice_parameter_a", "Å"),
        ("_cell_length_b", "lattice_parameter_b", "Å"),
        ("_cell_length_c", "lattice_parameter_c", "Å"),
        ("_cell_angle_alpha", "lattice_angle_alpha", "degrees"),
        ("_cell_angle_beta", "lattice_angle_beta", "degrees"),
        ("_cell_angle_gamma", "lattice_angle_gamma", "degrees"),
        ("_cell_volume", "unit_cell_volume", "Å³"),
    ]:
        field_value = _cif_value(cif_text, field)

        if field_value:
            properties.append(
                ChemicalProperty(
                    name,
                    float(field_value),
                    unit,
                    "SLICES reconstructed CIF",
                )
            )

    source_id = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return ChemicalRecord(
        record_id=f"slices:{source_id}",
        domain="inorganic",
        entity_kind="crystal",
        source="SLICES converter",
        source_id=source_id,
        preferred_name=formula,
        formula=formula,
        representations={"slices": value},
        properties=properties,
        source_url=SLICES_URL,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        license_name="LGPL-2.1",
        metadata={
            "space_group": _cif_value(
                cif_text,
                "_symmetry_space_group_name_H-M",
            ),
            "space_group_number": _cif_value(
                cif_text,
                "_symmetry_Int_Tables_number",
            ),
            "conversion_status": str(status),
        },
    )

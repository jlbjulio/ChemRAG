from dataclasses import asdict, dataclass, field
import math
from typing import Any, Literal


ChemicalDomain = Literal["organic", "inorganic"]
EntityKind = Literal["molecule", "crystal"]
PropertyValue = str | int | float | bool


@dataclass(frozen=True)
class ChemicalProperty:
    name: str
    value: PropertyValue
    unit: str | None = None
    method: str | None = None


@dataclass(frozen=True)
class ChemicalRecord:
    record_id: str
    domain: ChemicalDomain
    entity_kind: EntityKind
    source: str
    source_id: str
    preferred_name: str
    formula: str | None
    representations: dict[str, str]
    properties: list[ChemicalProperty]
    source_url: str
    retrieved_at: str
    license_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        required_text = {
            "record_id": self.record_id,
            "source": self.source,
            "source_id": self.source_id,
            "preferred_name": self.preferred_name,
            "source_url": self.source_url,
            "retrieved_at": self.retrieved_at,
        }

        for field_name, value in required_text.items():
            if not value.strip():
                raise ValueError(
                    f"The {field_name} field cannot be empty."
                )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


VISIBLE_REPRESENTATIONS = {
    "canonical_smiles",
    "isomeric_smiles",
    "inchi",
    "inchikey",
    "slices",
}

PROPERTY_LABELS = {
    "distance_from_convex_hull": (
        "Thermodynamic stability (distance from convex hull)"
    ),
    "formation_energy_per_atom": "Formation energy per atom",
    "unit_cell_volume": "Unit-cell volume",
    "volume_per_atom": "Volume per atom",
    "polar_surface_area": "Polar surface area",
}

def _format_value(value: PropertyValue) -> str:
    if isinstance(value, float) and math.isfinite(value):
        return f"{value:.6g}"

    return str(value)


def _method_for_context(method: str) -> tuple[str, str]:
    normalized = method.casefold()

    if "pbe" in normalized:
        return "calculated", "PBE calculation"

    if "dft" in normalized:
        return "calculated", "DFT calculation"

    if "calculated" in normalized or "descriptor" in normalized:
        return "calculated", "calculated descriptor"

    if "crystal structure record" in normalized:
        return "reported crystal data", "crystallographic record"

    return "reported value", "reported record"


def render_record_as_text(record: ChemicalRecord) -> str:
    lines = [
        f"CHEMICAL RECORD: {record.preferred_name}",
        f"Domain: {record.domain}",
        f"Entity type: {record.entity_kind}",
        f"Source: {record.source}",
        f"Source identifier: {record.source_id}",
    ]

    if record.formula:
        lines.append(f"Formula: {record.formula}")

    visible_representations = {
        name: value
        for name, value in record.representations.items()
        if name in VISIBLE_REPRESENTATIONS
    }

    if visible_representations:
        lines.append("Representations:")

        for name, value in sorted(visible_representations.items()):
            lines.append(f"- {name}: {value}")

    if record.properties:
        lines.append("Properties:")

        for item in record.properties:
            value = _format_value(item.value)

            if item.unit:
                value = f"{value} {item.unit}"

            if item.method:
                evidence_type, method = _method_for_context(item.method)
                value = (
                    f"{value} (evidence type: {evidence_type}; "
                    f"method: {method})"
                )

            label = PROPERTY_LABELS.get(
                item.name,
                item.name.replace("_", " ").capitalize(),
            )
            lines.append(f"- {label}: {value}")

    space_group = record.metadata.get("space_group")

    if space_group:
        lines.append(
            "Space group (evidence type: reported crystal data): "
            f"{space_group}"
        )

    elements = record.metadata.get("elements")

    if elements:
        lines.append(f"Elements: {', '.join(elements)}")

    metadata_labels = {
        "space_group_number": "Space group number",
        "layer_group": "Layer group",
        "magnetic": "Magnetic",
        "dimensionality": "Dimensionality",
        "number_of_elements": "Number of elements",
        "number_of_sites": "Number of sites",
        "prototype": "Prototype (evidence type: reported crystal data)",
        "publication_year": "Publication year",
        "doi": "DOI",
        "dataset": "Dataset",
    }

    for key, label in metadata_labels.items():
        value = record.metadata.get(key)

        if (
            value is not None
            and value != ""
            and value != []
        ):
            lines.append(f"{label}: {value}")

    return "\n".join(lines)

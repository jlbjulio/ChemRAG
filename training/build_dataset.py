import argparse
import json
import math
import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "training" / "data"
TRAIN_PATH = DATA_DIR / "train.jsonl"
VALIDATION_PATH = DATA_DIR / "validation.jsonl"

sys.path.insert(0, str(SRC_DIR))

from chemistry.chembl import fetch_chembl_molecule_page  # noqa: E402
from chemistry.cod import search_cod_by_formula  # noqa: E402
from chemistry.http_client import ChemistryApiError  # noqa: E402
from chemistry.oqmd import fetch_oqmd_material_page  # noqa: E402
from chemistry.oqmd_local import (  # noqa: E402
    fetch_local_oqmd_page,
    local_oqmd_available,
)
from chemistry.schema import (  # noqa: E402
    ChemicalProperty,
    ChemicalRecord,
)
from local_llm import NO_ANSWER, SYSTEM_PROMPT  # noqa: E402


PROPERTY_QUESTIONS = {
    "molecular_weight": "What is the molecular weight of {name}?",
    "alogp": "What is the ALogP value reported for {name}?",
    "xlogp": "What is the XLogP value reported for {name}?",
    "polar_surface_area": "What is the polar surface area of {name}?",
    "tpsa": "What is the TPSA of {name}?",
    "hydrogen_bond_acceptor_count": (
        "How many hydrogen-bond acceptors does {name} have?"
    ),
    "hydrogen_bond_donor_count": (
        "How many hydrogen-bond donors does {name} have?"
    ),
    "rotatable_bond_count": "How many rotatable bonds does {name} have?",
    "energy_per_atom": "What energy per atom is reported for {name}?",
    "band_gap": "What band gap is reported for {name}?",
    "formation_energy_per_atom": (
        "What is the formation energy per atom of {name}?"
    ),
    "distance_from_convex_hull": (
        "What is the distance from the convex hull for {name}?"
    ),
    "lattice_parameter_a": "What lattice parameter a is reported for {name}?",
    "lattice_parameter_b": "What lattice parameter b is reported for {name}?",
    "lattice_parameter_c": "What lattice parameter c is reported for {name}?",
    "lattice_angle_alpha": "What lattice angle alpha is reported for {name}?",
    "lattice_angle_beta": "What lattice angle beta is reported for {name}?",
    "lattice_angle_gamma": "What lattice angle gamma is reported for {name}?",
    "unit_cell_volume": "What unit-cell volume is reported for {name}?",
    "volume_per_atom": "What volume per atom is reported for {name}?",
    "magnetization_per_atom": (
        "What magnetization per atom is reported for {name}?"
    ),
    "atomic_volume_per_atom": (
        "What atomic volume per atom is reported for {name}?"
    ),
    "volume_deviation": "What volume deviation is reported for {name}?",
}

COD_SEED_FORMULAS = [
    "BaTiO3", "SiO2", "Al2O3", "Fe2O3", "TiO2", "ZnO", "MgO",
    "CaCO3", "NaCl", "LiFePO4", "SrTiO3", "ZrO2", "HfO2", "CuO",
    "Cu2O", "NiO", "CoO", "MnO2", "WO3", "MoS2", "WS2", "GaN",
    "SiC", "BN", "CdS", "PbS", "CsPbBr3", "KTaO3", "KNbO3",
    "LiCoO2", "LiMn2O4", "Na3V2PO4", "YBa2Cu3O7", "BiFeO3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a balanced chemistry instruction dataset from live "
            "ChEMBL and OQMD records without keeping the bulk databases."
        )
    )
    parser.add_argument(
        "--target-examples",
        type=int,
        default=10_000,
        help="Approximate total number of examples (default: 10000).",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="API records requested per page (default: 100).",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1,
        help="Fraction of source records reserved for validation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _format_property(item: ChemicalProperty) -> str:
    value = str(item.value)

    if item.unit:
        value = f"{value} {item.unit}"

    return value


def _property_label(name: str) -> str:
    labels = {
        "distance_from_convex_hull": (
            "thermodynamic stability (distance from the convex hull)"
        ),
        "formation_energy_per_atom": "formation energy per atom",
        "unit_cell_volume": "unit-cell volume",
        "volume_per_atom": "volume per atom",
        "polar_surface_area": "polar surface area",
    }
    return labels.get(name, name.replace("_", " "))


def _public_method(method: str | None) -> str:
    """Describe the calculation type without teaching provider citations."""
    if not method:
        return ""

    normalized = method.casefold()

    if "pbe" in normalized:
        return " using a PBE calculation"

    if "dft" in normalized:
        return " using a DFT calculation"

    if "crystal structure record" in normalized:
        return " from reported crystal data"

    if "calculated" in normalized or "descriptor" in normalized:
        return " as a calculated value"

    return ""


def _spanish_property_label(name: str) -> str:
    labels = {
        "molecular_weight": "peso molecular",
        "polar_surface_area": "área de superficie polar",
        "hydrogen_bond_acceptor_count": "aceptores de enlaces de hidrógeno",
        "hydrogen_bond_donor_count": "donantes de enlaces de hidrógeno",
        "rotatable_bond_count": "enlaces rotables",
        "energy_per_atom": "energía por átomo",
        "band_gap": "brecha de banda",
        "formation_energy_per_atom": "energía de formación por átomo",
        "distance_from_convex_hull": "distancia al casco convexo",
        "lattice_parameter_a": "parámetro de red a",
        "lattice_parameter_b": "parámetro de red b",
        "lattice_parameter_c": "parámetro de red c",
        "lattice_angle_alpha": "ángulo de red alfa",
        "lattice_angle_beta": "ángulo de red beta",
        "lattice_angle_gamma": "ángulo de red gamma",
        "unit_cell_volume": "volumen de la celda unitaria",
        "volume_per_atom": "volumen por átomo",
        "magnetization_per_atom": "magnetización por átomo",
        "atomic_volume_per_atom": "volumen atómico por átomo",
        "volume_deviation": "desviación de volumen",
    }
    return labels.get(name, name.replace("_", " "))


def _spanish_public_method(method: str | None) -> str:
    if not method:
        return ""

    normalized = method.casefold()

    if "pbe" in normalized:
        return " mediante un cálculo PBE"

    if "dft" in normalized:
        return " mediante un cálculo DFT"

    if "crystal structure record" in normalized:
        return " según datos cristalográficos reportados"

    if "calculated" in normalized or "descriptor" in normalized:
        return " como valor calculado"

    return ""


def _join_labels(labels: list[str]) -> str:
    if len(labels) == 1:
        return labels[0]

    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"

    return f"{', '.join(labels[:-1])}, and {labels[-1]}"


def _example(
    context: str,
    question: str,
    answer: str,
) -> dict:
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Retrieved context:\n{context}\n\n"
                    f"Question:\n{question}"
                ),
            },
        ],
        "completion": [{"role": "assistant", "content": answer}],
    }


def _compact_context(
    record: ChemicalRecord,
    facts: list[str] | None = None,
) -> str:
    lines = [
        f"Chemical entity: {record.preferred_name}",
        f"Domain: {record.domain}",
        f"Source: {record.source}",
        f"Source identifier: {record.source_id}",
    ]

    if record.formula:
        lines.append(f"Formula: {record.formula}")

    if facts:
        lines.extend(facts)

    return "\n".join(lines)


def examples_from_record(record: ChemicalRecord) -> list[dict]:
    base_context = _compact_context(record)
    examples = []

    if record.formula:
        examples.append(
            _example(
                base_context,
                f"What molecular or chemical formula is reported for "
                f"{record.preferred_name}?",
                f"The reported formula is {record.formula}.",
            )
        )

    smiles = record.representations.get("canonical_smiles")

    if smiles and len(smiles) <= 120:
        examples.append(
            _example(
                _compact_context(
                    record,
                    [f"Canonical SMILES: {smiles}"],
                ),
                f"What canonical SMILES is reported for "
                f"{record.preferred_name}?",
                f"The canonical SMILES is {smiles}.",
            )
        )

    for item in record.properties:
        question_template = PROPERTY_QUESTIONS.get(item.name)

        if not question_template:
            continue

        value = _format_property(item)
        method = _public_method(item.method)
        examples.append(
            _example(
                _compact_context(
                    record,
                    [
                        f"Reported {item.name}: {value}"
                        + (
                            f" (method: {item.method})"
                            if item.method
                            else ""
                        )
                    ],
                ),
                question_template.format(name=record.preferred_name),
                f"The reported {_property_label(item.name)} is {value}"
                f"{method}.",
            )
        )

    space_group = record.metadata.get("space_group")

    if space_group:
        examples.append(
            _example(
                _compact_context(
                    record,
                    [f"Space group: {space_group}"],
                ),
                f"What space group is reported for {record.preferred_name}?",
                f"The reported space group is {space_group}.",
            )
        )

    metadata_questions = {
        "space_group_number": "What space-group number is reported for {name}?",
        "number_of_sites": "How many atomic sites are reported for {name}?",
        "layer_group": "What layer group is reported for {name}?",
        "magnetic": "Is {name} reported as magnetic?",
    }

    for key, question_template in metadata_questions.items():
        value = record.metadata.get(key)

        if value is not None and value != "":
            examples.append(
                _example(
                    _compact_context(
                        record,
                        [f"Reported {key}: {value}"],
                    ),
                    question_template.format(name=record.preferred_name),
                    f"The reported {key.replace('_', ' ')} is {value}.",
                )
            )

    missing_question = (
        f"What clinical dosage is recommended for {record.preferred_name}?"
        if record.domain == "organic"
        else (
            f"What synthesis temperature was used for "
            f"{record.preferred_name}?"
        )
    )
    available_facts = [
        f"Reported {item.name}: {_format_property(item)}"
        for item in record.properties
    ]
    available_facts.extend(
        f"Reported {key}: {value}"
        for key, value in record.metadata.items()
        if key in metadata_questions and value is not None and value != ""
    )

    if smiles and len(smiles) <= 120:
        available_facts.append(f"Canonical SMILES: {smiles}")

    if space_group:
        available_facts.append(f"Space group: {space_group}")

    candidates: list[tuple[str, str, str]] = []

    if record.formula:
        candidates.append(
            (
                "formula",
                f"Formula: {record.formula}",
                f"The reported formula is {record.formula}.",
            )
        )

    for item in record.properties:
        if item.name not in PROPERTY_QUESTIONS:
            continue

        label = _property_label(item.name)
        value = _format_property(item)
        fact = f"Reported {item.name}: {value}"

        if item.method:
            fact += f" (method: {item.method})"

        candidates.append(
            (
                label,
                fact,
                f"The reported {label} is {value}"
                f"{_public_method(item.method)}.",
            )
        )

    if space_group:
        candidates.append(
            (
                "space group",
                f"Space group: {space_group}",
                f"The reported space group is {space_group}.",
            )
        )

    prototype = record.metadata.get("prototype")

    if prototype:
        candidates.append(
            (
                "crystal structure or prototype",
                f"Prototype: {prototype}",
                f"The reported crystal prototype is {prototype}.",
            )
        )

    for key in metadata_questions:
        value = record.metadata.get(key)

        if value is not None and value != "":
            label = key.replace("_", " ")
            candidates.append(
                (
                    label,
                    f"Reported {key}: {value}",
                    f"The reported {label} is {value}.",
                )
            )

    if len(candidates) >= 2:
        requested = candidates[::2][:3]

        if len(requested) < 2:
            requested = candidates[:2]

        context_facts = [candidate[1] for candidate in candidates[:5]]
        examples.append(
            _example(
                _compact_context(record, context_facts),
                f"Report only the {_join_labels([item[0] for item in requested])} "
                f"for {record.preferred_name}.",
                "\n".join(
                    f"- {item[0].capitalize()}: {item[2]}"
                    for item in requested
                ),
            )
        )

        available = candidates[1] if len(candidates) > 1 else candidates[0]
        unavailable_label = (
            "clinical dosage"
            if record.domain == "organic"
            else "synthesis temperature"
        )
        examples.append(
            _example(
                _compact_context(record, context_facts),
                f"Report the {available[0]} and {unavailable_label} for "
                f"{record.preferred_name}.",
                f"- {available[0].capitalize()}: {available[2]}\n"
                f"- {unavailable_label.capitalize()}: Not available in "
                "the retrieved evidence.",
            )
        )

    examples.append(
        _example(
            _compact_context(record, available_facts[:4]),
            missing_question,
            NO_ANSWER,
        )
    )
    spanish_examples = []

    if record.formula:
        spanish_examples.append(
            _example(
                base_context,
                f"¿Qué fórmula química se reporta para "
                f"{record.preferred_name}?",
                f"La fórmula reportada es {record.formula}.",
            )
        )

    for item in record.properties:
        if item.name not in PROPERTY_QUESTIONS:
            continue

        label = _spanish_property_label(item.name)
        value = _format_property(item)
        spanish_examples.append(
            _example(
                _compact_context(
                    record,
                    [
                        f"Reported {item.name}: {value}"
                        + (
                            f" (method: {item.method})"
                            if item.method
                            else ""
                        )
                    ],
                ),
                f"¿Cuál es el valor reportado de {label} para "
                f"{record.preferred_name}?",
                f"El valor reportado de {label} es {value}"
                f"{_spanish_public_method(item.method)}.",
            )
        )

    if space_group:
        spanish_examples.append(
            _example(
                _compact_context(record, [f"Space group: {space_group}"]),
                f"¿Cuál es el grupo espacial reportado para "
                f"{record.preferred_name}?",
                f"El grupo espacial reportado es {space_group}.",
            )
        )

    spanish_missing_question = (
        f"¿Qué dosis clínica se recomienda para {record.preferred_name}?"
        if record.domain == "organic"
        else (
            f"¿Qué temperatura de síntesis se usó para "
            f"{record.preferred_name}?"
        )
    )
    spanish_examples.append(
        _example(
            _compact_context(record, available_facts[:4]),
            spanish_missing_question,
            NO_ANSWER,
        )
    )
    return [*examples, *spanish_examples]


def collect_records(
    organic_target: int,
    inorganic_target: int,
    page_size: int,
) -> tuple[list[ChemicalRecord], list[ChemicalRecord]]:
    organic_records = []
    inorganic_records = []
    organic_offset = 0
    inorganic_offset = 0
    failures = {"ChEMBL": 0, "OQMD": 0}
    oqmd_available = True
    cod_formula_index = 0
    inorganic_by_id: dict[str, ChemicalRecord] = {}

    while (
        len(organic_records) < organic_target
        or len(inorganic_records) < inorganic_target
    ):
        if len(organic_records) < organic_target:
            try:
                page = fetch_chembl_molecule_page(
                    organic_offset,
                    page_size,
                )
                organic_records.extend(page)
                organic_offset += page_size
                failures["ChEMBL"] = 0
                print(
                    f"ChEMBL records: {len(organic_records)}/"
                    f"{organic_target}",
                    flush=True,
                )
            except (ChemistryApiError, ValueError) as error:
                failures["ChEMBL"] += 1
                print(f"ChEMBL warning: {error}", flush=True)

        if len(inorganic_records) < inorganic_target and oqmd_available:
            try:
                page = fetch_oqmd_material_page(
                    inorganic_offset,
                    page_size,
                )
                inorganic_records.extend(page)
                inorganic_by_id.update(
                    {record.record_id: record for record in page}
                )
                inorganic_offset += page_size
                failures["OQMD"] = 0
                print(
                    f"OQMD records: {len(inorganic_records)}/"
                    f"{inorganic_target}",
                    flush=True,
                )
            except (ChemistryApiError, ValueError) as error:
                failures["OQMD"] += 1
                print(f"OQMD warning: {error}", flush=True)
                fallback_name = (
                    "the local OQMD dataset"
                    if local_oqmd_available()
                    else "COD records"
                )
                print(
                    f"Switching this dataset build to {fallback_name}.",
                    flush=True,
                )
                oqmd_available = False

        if len(inorganic_records) < inorganic_target and not oqmd_available:
            if local_oqmd_available():
                page = fetch_local_oqmd_page(
                    inorganic_offset,
                    page_size,
                )
                inorganic_offset += page_size
                inorganic_by_id.update(
                    {record.record_id: record for record in page}
                )
                inorganic_records = list(inorganic_by_id.values())
                print(
                    f"Local OQMD records: {len(inorganic_records)}/"
                    f"{inorganic_target}",
                    flush=True,
                )
                continue

            if cod_formula_index >= len(COD_SEED_FORMULAS):
                raise RuntimeError(
                    "OQMD is unavailable and the COD fallback formulas did "
                    "not provide enough distinct inorganic records. Run the "
                    "command again when OQMD is available or request fewer "
                    "examples."
                )

            formula = COD_SEED_FORMULAS[cod_formula_index]
            cod_formula_index += 1

            try:
                page = search_cod_by_formula(formula, limit=50)
                inorganic_by_id.update(
                    {record.record_id: record for record in page}
                )
                inorganic_records = list(inorganic_by_id.values())
                print(
                    f"COD records: {len(inorganic_records)}/"
                    f"{inorganic_target}",
                    flush=True,
                )
            except (ChemistryApiError, ValueError) as error:
                print(f"COD warning ({formula}): {error}", flush=True)

        if failures["ChEMBL"] >= 3:
            raise RuntimeError(
                "A source failed three consecutive times. Run the command "
                "again later; no partial dataset was written."
            )

    return (
        organic_records[:organic_target],
        inorganic_records[:inorganic_target],
    )


def split_records(
    records: list[ChemicalRecord],
    validation_ratio: float,
) -> tuple[list[ChemicalRecord], list[ChemicalRecord]]:
    validation_size = max(1, round(len(records) * validation_ratio))
    return records[validation_size:], records[:validation_size]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False))
            file.write("\n")

    temporary_path.replace(path)


def take_examples(
    records: list[ChemicalRecord],
    target: int,
    random_generator: random.Random,
    split_name: str,
) -> list[dict]:
    examples = [
        example
        for record in records
        for example in examples_from_record(record)
    ]
    random_generator.shuffle(examples)

    if len(examples) < target:
        raise RuntimeError(
            f"The {split_name} split produced {len(examples)} examples, "
            f"but {target} are required. Request a smaller dataset or add "
            "more source records."
        )

    return examples[:target]


def main() -> None:
    args = parse_args()

    if args.target_examples < 100:
        raise ValueError("target-examples must be at least 100.")

    if not 0 < args.validation_ratio < 0.5:
        raise ValueError("validation-ratio must be between 0 and 0.5.")

    random_generator = random.Random(args.seed)
    examples_per_domain = args.target_examples // 2
    organic_record_target = math.ceil(examples_per_domain / 6)
    inorganic_record_target = math.ceil(examples_per_domain / 9)
    organic_records, inorganic_records = collect_records(
        organic_record_target,
        inorganic_record_target,
        args.page_size,
    )
    random_generator.shuffle(organic_records)
    random_generator.shuffle(inorganic_records)
    organic_train, organic_validation = split_records(
        organic_records,
        args.validation_ratio,
    )
    inorganic_train, inorganic_validation = split_records(
        inorganic_records,
        args.validation_ratio,
    )
    validation_target = round(
        args.target_examples * args.validation_ratio
    )
    training_target = args.target_examples - validation_target
    organic_training_target = training_target // 2
    organic_validation_target = validation_target // 2
    train_examples = take_examples(
        organic_train,
        organic_training_target,
        random_generator,
        "organic training",
    ) + take_examples(
        inorganic_train,
        training_target - organic_training_target,
        random_generator,
        "inorganic training",
    )
    validation_examples = take_examples(
        organic_validation,
        organic_validation_target,
        random_generator,
        "organic validation",
    ) + take_examples(
        inorganic_validation,
        validation_target - organic_validation_target,
        random_generator,
        "inorganic validation",
    )
    random_generator.shuffle(train_examples)
    random_generator.shuffle(validation_examples)
    write_jsonl(TRAIN_PATH, train_examples)
    write_jsonl(VALIDATION_PATH, validation_examples)

    print("\nDataset created from live scientific records.")
    print(f"Training examples: {len(train_examples)}")
    print(f"Validation examples: {len(validation_examples)}")
    print(f"Training file: {TRAIN_PATH}")
    print(f"Validation file: {VALIDATION_PATH}")
    print(
        "Source records were split before example generation to reduce "
        "entity leakage between training and validation."
    )


if __name__ == "__main__":
    main()

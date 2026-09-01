import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from .chembl import search_chembl_molecules
from .c2db import search_c2db_by_formula
from .cod import search_cod_by_formula
from .http_client import ChemistryApiError
from .oqmd import fetch_oqmd_materials
from .oqmd_local import (
    local_oqmd_available,
    search_local_oqmd_by_formula,
)
from .pubchem import fetch_pubchem_compound
from .schema import ChemicalRecord
from .slices_service import decode_slices
from .smiles_tools import analyze_smiles, is_valid_smiles


QueryKind = Literal[
    "compound_name",
    "smiles",
    "pubchem_cid",
    "inchikey",
    "formula",
    "slices",
    "general",
]

ELEMENTS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
}
FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*(?:\.\d+)?)")
PREFIX_PATTERN = re.compile(
    r"^\s*(compound|name|smiles|cid|inchikey|formula|material|slices)"
    r"\s*:\s*(.+?)\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QueryIntent:
    kind: QueryKind
    value: str


@dataclass(frozen=True)
class OnlineRetrieval:
    intent: QueryIntent
    records: tuple[ChemicalRecord, ...]
    warnings: tuple[str, ...]


def _is_formula(value: str) -> bool:
    matches = list(FORMULA_TOKEN.finditer(value))

    if not matches:
        return False

    if "".join(match.group(0) for match in matches) != value:
        return False

    elements = [match.group(1) for match in matches]

    return (
        all(element in ELEMENTS for element in elements)
        and (len(elements) > 1 or any(character.isdigit() for character in value))
    )


def _formula_from_question(question: str) -> str | None:
    for candidate in re.findall(
        r"\b(?:[A-Z][a-z]?\d*(?:\.\d+)?){1,8}\b",
        question,
    ):
        if _is_formula(candidate):
            return candidate

    return None


def _compound_from_question(question: str) -> str | None:
    cleaned = question.strip().rstrip("?.!")
    patterns = [
        r"\b(?:of|about|for)\s+(?:the\s+)?([A-Za-z][A-Za-z0-9 -]{1,60})$",
        r"\b(?:de|del|sobre)\s+(?:la\s+|el\s+)?([A-Za-z\u00c0-\u00ff][A-Za-z\u00c0-\u00ff0-9 -]{1,60})$",
        r"\bwhat is\s+([A-Za-z][A-Za-z0-9 -]{1,40})['\u2019]s\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, cleaned, re.IGNORECASE)

        if match:
            return match.group(1).strip()

    words = cleaned.split()

    if len(words) == 1 and all(
        re.fullmatch(r"[A-Za-z\u00c0-\u00ff0-9-]+", word)
        for word in words
    ):
        return cleaned

    return None


def detect_query_intent(question: str) -> QueryIntent:
    prefix_match = PREFIX_PATTERN.match(question)

    if prefix_match:
        prefix, value = prefix_match.groups()
        normalized_prefix = prefix.casefold()
        kind_by_prefix: dict[str, QueryKind] = {
            "compound": "compound_name",
            "name": "compound_name",
            "smiles": "smiles",
            "cid": "pubchem_cid",
            "inchikey": "inchikey",
            "formula": "formula",
            "material": "formula",
            "slices": "slices",
        }
        return QueryIntent(kind_by_prefix[normalized_prefix], value.strip())

    stripped_question = question.strip()

    if is_valid_smiles(stripped_question):
        return QueryIntent("smiles", stripped_question)

    formula = _formula_from_question(question)

    if formula:
        return QueryIntent("formula", formula)

    compound = _compound_from_question(question)

    if compound:
        return QueryIntent("compound_name", compound)

    return QueryIntent("general", question.strip())


def _deduplicate_records(
    records: list[ChemicalRecord],
) -> tuple[ChemicalRecord, ...]:
    unique_records = {}

    for record in records:
        unique_records[record.record_id] = record

    return tuple(unique_records.values())


@lru_cache(maxsize=128)
def _retrieve_cached(kind: QueryKind, value: str) -> OnlineRetrieval:
    intent = QueryIntent(kind, value)

    if kind == "general":
        return OnlineRetrieval(intent, (), ())

    if kind == "slices":
        try:
            slices_record = decode_slices(value)
            formula_result = _retrieve_cached(
                "formula",
                slices_record.formula or slices_record.preferred_name,
            )
            return OnlineRetrieval(
                intent,
                _deduplicate_records(
                    [slices_record, *formula_result.records]
                ),
                formula_result.warnings,
            )
        except (ChemistryApiError, OSError, ValueError) as error:
            return OnlineRetrieval(intent, (), (str(error),))

    records: list[ChemicalRecord] = []
    warnings = []

    if kind == "formula":
        oqmd_failed = False

        with ThreadPoolExecutor(max_workers=3) as executor:
            tasks = {
                executor.submit(fetch_oqmd_materials, value, 5): "OQMD",
                executor.submit(search_cod_by_formula, value, 5): "COD",
                executor.submit(search_c2db_by_formula, value, 5): "C2DB",
            }

            for task in as_completed(tasks):
                try:
                    records.extend(task.result())
                except (ChemistryApiError, ValueError) as error:
                    warnings.append(str(error))

                    if tasks[task] == "OQMD":
                        oqmd_failed = True

        if not records and oqmd_failed and local_oqmd_available():
            try:
                local_records = search_local_oqmd_by_formula(value, 5)

                if local_records:
                    records.extend(local_records)
                    warnings.append(
                        "OQMD online was unavailable; local OQMD fallback "
                        "records were used."
                    )
            except (OSError, sqlite3.Error, ValueError) as error:
                warnings.append(f"Local OQMD fallback failed: {error}")

        return OnlineRetrieval(
            intent,
            _deduplicate_records(records),
            tuple(warnings),
        )

    namespace_by_kind = {
        "compound_name": "name",
        "smiles": "smiles",
        "pubchem_cid": "cid",
        "inchikey": "inchikey",
    }
    tasks = []

    if kind == "smiles":
        try:
            records.append(analyze_smiles(value))
        except ValueError as error:
            warnings.append(str(error))

    with ThreadPoolExecutor(max_workers=2) as executor:
        tasks.append(
            executor.submit(
                fetch_pubchem_compound,
                value,
                namespace_by_kind[kind],
            )
        )

        if kind == "compound_name":
            tasks.append(
                executor.submit(search_chembl_molecules, value, 3)
            )

        for task in as_completed(tasks):
            try:
                result = task.result()

                if isinstance(result, list):
                    records.extend(result)
                else:
                    records.append(result)
            except (ChemistryApiError, ValueError) as error:
                warnings.append(str(error))

    return OnlineRetrieval(
        intent,
        _deduplicate_records(records),
        tuple(warnings),
    )


def retrieve_online(question: str) -> OnlineRetrieval:
    intent = detect_query_intent(question)
    return _retrieve_cached(intent.kind, intent.value)


def clear_online_cache() -> None:
    _retrieve_cached.cache_clear()

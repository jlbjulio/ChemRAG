import hashlib
from collections.abc import Callable
from datetime import datetime, timezone
from typing import cast

from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

from .schema import ChemicalProperty, ChemicalRecord


_disable_log = cast(
    Callable[[str], None],
    getattr(RDLogger, "DisableLog"),
)
_molecular_weight = cast(
    Callable[[Chem.Mol], float],
    getattr(Descriptors, "MolWt"),
)
_molecular_logp = cast(
    Callable[[Chem.Mol], float],
    getattr(Crippen, "MolLogP"),
)
_hydrogen_acceptors = cast(
    Callable[[Chem.Mol], int],
    getattr(Lipinski, "NumHAcceptors"),
)
_hydrogen_donors = cast(
    Callable[[Chem.Mol], int],
    getattr(Lipinski, "NumHDonors"),
)
_rotatable_bonds = cast(
    Callable[[Chem.Mol], int],
    getattr(Lipinski, "NumRotatableBonds"),
)
_ring_count = cast(
    Callable[[Chem.Mol], int],
    getattr(Lipinski, "RingCount"),
)

_disable_log("rdApp.error")


def molecule_from_smiles(smiles: str) -> Chem.Mol | None:
    value = smiles.strip()

    if not value or any(character.isspace() for character in value):
        return None

    return Chem.MolFromSmiles(value)


def is_valid_smiles(smiles: str) -> bool:
    return molecule_from_smiles(smiles) is not None


def analyze_smiles(smiles: str) -> ChemicalRecord:
    molecule = molecule_from_smiles(smiles)

    if molecule is None:
        raise ValueError("RDKit could not parse the supplied SMILES string.")

    canonical_smiles = Chem.MolToSmiles(molecule, canonical=True)
    formula = rdMolDescriptors.CalcMolFormula(molecule)
    properties = [
        ChemicalProperty(
            "molecular_weight",
            round(_molecular_weight(molecule), 4),
            "g/mol",
            "RDKit descriptor",
        ),
        ChemicalProperty(
            "alogp",
            round(_molecular_logp(molecule), 4),
            method="RDKit Crippen descriptor",
        ),
        ChemicalProperty(
            "tpsa",
            round(rdMolDescriptors.CalcTPSA(molecule), 4),
            "Å²",
            "RDKit descriptor",
        ),
        ChemicalProperty(
            "hydrogen_bond_acceptor_count",
            _hydrogen_acceptors(molecule),
            method="RDKit Lipinski descriptor",
        ),
        ChemicalProperty(
            "hydrogen_bond_donor_count",
            _hydrogen_donors(molecule),
            method="RDKit Lipinski descriptor",
        ),
        ChemicalProperty(
            "rotatable_bond_count",
            _rotatable_bonds(molecule),
            method="RDKit Lipinski descriptor",
        ),
        ChemicalProperty(
            "ring_count",
            _ring_count(molecule),
            method="RDKit descriptor",
        ),
    ]
    source_id = hashlib.sha256(
        canonical_smiles.encode("utf-8")
    ).hexdigest()[:12]
    representations = {"canonical_smiles": canonical_smiles}

    try:
        representations["inchikey"] = Chem.MolToInchiKey(molecule)
    except RuntimeError:
        pass

    return ChemicalRecord(
        record_id=f"rdkit:{source_id}",
        domain="organic",
        entity_kind="molecule",
        source="RDKit",
        source_id=source_id,
        preferred_name=formula,
        formula=formula,
        representations=representations,
        properties=properties,
        source_url="https://www.rdkit.org/docs/",
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        license_name="BSD-3-Clause",
        metadata={"calculation": "local 2D molecular descriptors"},
    )

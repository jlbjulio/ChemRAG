import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chemistry.http_client import ChemistryApiError  # noqa: E402
from chemistry.online import (  # noqa: E402
    clear_online_cache,
    detect_query_intent,
    retrieve_online,
)
from chemistry.oqmd_local import (  # noqa: E402
    local_oqmd_available,
    search_local_oqmd_by_formula,
)
from chemistry.smiles_tools import analyze_smiles  # noqa: E402
from chemistry.schema import render_record_as_text  # noqa: E402
from local_llm import LocalLLM  # noqa: E402
from search_chunks import (  # noqa: E402
    NO_ANSWER,
    build_context,
    enforce_structured_scope,
)
from training.build_dataset import examples_from_record  # noqa: E402


class QueryRoutingTests(unittest.TestCase):
    def test_compound_question(self) -> None:
        intent = detect_query_intent(
            "What is the molecular weight of aspirin?"
        )
        self.assertEqual(intent.kind, "compound_name")
        self.assertEqual(intent.value, "aspirin")

    def test_formula_prefix(self) -> None:
        intent = detect_query_intent("formula: BaTiO3")
        self.assertEqual(intent.kind, "formula")
        self.assertEqual(intent.value, "BaTiO3")

    def test_bare_smiles(self) -> None:
        intent = detect_query_intent("CC(=O)OC1=CC=CC=C1C(=O)O")
        self.assertEqual(intent.kind, "smiles")

    def test_general_question(self) -> None:
        intent = detect_query_intent("Explain covalent bonding.")
        self.assertEqual(intent.kind, "general")


class ChemistryToolTests(unittest.TestCase):
    def test_rdkit_aspirin_formula(self) -> None:
        record = analyze_smiles("CC(=O)OC1=CC=CC=C1C(=O)O")
        self.assertEqual(record.formula, "C9H8O4")

    @unittest.skipUnless(
        local_oqmd_available(),
        "The optional local OQMD dataset is not installed.",
    )
    def test_local_oqmd_normalizes_formula_order(self) -> None:
        records = search_local_oqmd_by_formula("BaO3Ti", 1)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].formula, "BaTiO3")

    def test_context_excludes_operational_metadata(self) -> None:
        record = analyze_smiles("CC(=O)OC1=CC=CC=C1C(=O)O")
        text = render_record_as_text(record)
        self.assertIn("Molecular weight", text)
        self.assertIn("Alogp", text)
        self.assertNotIn("Reported license", text)
        self.assertNotIn("Retrieved at", text)

    def test_direct_smiles_lookup_keeps_the_full_summary(self) -> None:
        smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
        record = analyze_smiles(smiles)
        text = render_record_as_text(record)
        self.assertIn("Molecular weight", text)
        self.assertIn("canonical_smiles", text)


class OnlineFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_online_cache()
        self.record = analyze_smiles("O")

    def tearDown(self) -> None:
        clear_online_cache()

    @patch("chemistry.online.search_local_oqmd_by_formula")
    @patch("chemistry.online.local_oqmd_available", return_value=True)
    @patch("chemistry.online.search_c2db_by_formula", return_value=[])
    @patch("chemistry.online.search_cod_by_formula", return_value=[])
    @patch("chemistry.online.fetch_oqmd_materials")
    def test_local_oqmd_is_not_used_when_online_succeeds(
        self,
        online_mock,
        _cod_mock,
        _c2db_mock,
        _available_mock,
        local_mock,
    ) -> None:
        online_mock.return_value = [self.record]
        result = retrieve_online("formula: H2O")
        self.assertEqual(len(result.records), 1)
        local_mock.assert_not_called()

    @patch("chemistry.online.search_local_oqmd_by_formula")
    @patch("chemistry.online.local_oqmd_available", return_value=True)
    @patch("chemistry.online.search_c2db_by_formula", return_value=[])
    @patch("chemistry.online.search_cod_by_formula", return_value=[])
    @patch("chemistry.online.fetch_oqmd_materials")
    def test_local_oqmd_is_used_when_online_fails(
        self,
        online_mock,
        _cod_mock,
        _c2db_mock,
        _available_mock,
        local_mock,
    ) -> None:
        online_mock.side_effect = ChemistryApiError("OQMD unavailable")
        local_mock.return_value = [self.record]
        result = retrieve_online("formula: H2O")
        self.assertEqual(len(result.records), 1)
        local_mock.assert_called_once_with("H2O", 5)
        self.assertTrue(
            any("local OQMD fallback" in warning for warning in result.warnings)
        )

    @patch("chemistry.online.search_local_oqmd_by_formula")
    @patch("chemistry.online.local_oqmd_available", return_value=True)
    @patch("chemistry.online.search_c2db_by_formula", return_value=[])
    @patch("chemistry.online.search_cod_by_formula")
    @patch("chemistry.online.fetch_oqmd_materials")
    def test_working_web_alternative_prevents_local_fallback(
        self,
        online_mock,
        cod_mock,
        _c2db_mock,
        _available_mock,
        local_mock,
    ) -> None:
        online_mock.side_effect = ChemistryApiError("OQMD unavailable")
        cod_mock.return_value = [self.record]
        result = retrieve_online("formula: H2O")
        self.assertEqual(len(result.records), 1)
        local_mock.assert_not_called()


class ContextAssemblyTests(unittest.TestCase):
    def test_context_contains_every_reranked_result(self) -> None:
        results = [
            (
                1.0,
                1.0,
                {
                    "source": f"document_{index}.txt",
                    "text": f"CHUNK {index}",
                },
            )
            for index in range(1, 19)
        ]
        context = build_context(results)
        self.assertIn("CHUNK 1", context)
        self.assertIn("CHUNK 18", context)
        self.assertEqual(context.count("Evidence [S"), 18)

    def test_structured_scope_uses_evidence_and_removes_extra_fields(self) -> None:
        question = (
            "Report the available crystal structure, space group, band gap, "
            "formation energy, and thermodynamic stability. Distinguish "
            "calculated values from experimental data."
        )
        context = (
            "Prototype (evidence type: reported crystal data): BaTiO3(tet)\n"
            "Space group (evidence type: reported crystal data): P 4 m m\n"
            "- Band gap: 1.9 eV (evidence type: calculated; method: DFT)\n"
            "- Formation energy per atom: -3.2 eV/atom "
            "(evidence type: calculated; method: DFT)\n"
            "- Unit-cell volume: 64.1 angstrom cubed"
        )
        candidate = "The structure is tetrahedral. Volume: 64.1."
        answer = enforce_structured_scope(question, context, candidate)
        self.assertIn("Crystal structure: BaTiO3(tet) (reported crystal data)", answer)
        self.assertIn("Space group: P 4 m m (reported crystal data)", answer)
        self.assertIn("Band gap: 1.9 eV (calculated)", answer)
        self.assertIn("Formation energy: -3.2 eV/atom (calculated)", answer)
        self.assertIn("Thermodynamic stability: Not available", answer)
        self.assertNotIn("64.1", answer)

    def test_structured_scope_refuses_when_every_field_is_missing(self) -> None:
        answer = enforce_structured_scope(
            "What are the band gap and formation energy?",
            "Formula: BaTiO3",
            "Invented answer",
        )
        self.assertEqual(answer, NO_ANSWER)


class FineTuningSetupTests(unittest.TestCase):
    def test_dataset_contains_spanish_instructions(self) -> None:
        record = analyze_smiles("CC(=O)OC1=CC=CC=C1C(=O)O")
        examples = examples_from_record(record)
        questions = [
            item["prompt"][1]["content"]
            for item in examples
        ]
        self.assertTrue(any("¿" in question for question in questions))

    @patch("local_llm.ADAPTER_DIR", Path("missing-test-adapter"))
    def test_chat_model_requires_the_lora_adapter(self) -> None:
        with self.assertRaises(FileNotFoundError):
            LocalLLM()

if __name__ == "__main__":
    unittest.main()

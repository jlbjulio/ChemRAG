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
from search_chunks import (  # noqa: E402
    enforce_requested_scope,
    split_into_batches,
    split_into_source_batches,
)


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


class BatchingTests(unittest.TestCase):
    def test_multi_item_scope_removes_unrequested_fields(self) -> None:
        question = (
            "Report the crystal structure, space group, band gap, and "
            "thermodynamic stability. Distinguish calculated values."
        )
        context = (
            "Space group (evidence type: reported crystal data): P 4 m m\n"
            "Band gap: 1.9 eV (evidence type: calculated; method: DFT)"
        )
        candidate = (
            "- Lattice parameter a: 3.9 Å\n"
            "- Space group: P 4 m m\n"
            "- Band gap: 1.9 eV\n"
            "- Thermodynamic stability: Not provided"
        )
        answer = enforce_requested_scope(question, context, candidate)
        self.assertNotIn("Lattice parameter", answer)
        self.assertIn("Crystal structure: Not available", answer)
        self.assertIn("Space group: P 4 m m (reported crystal data)", answer)
        self.assertIn("Band gap: 1.9 eV (calculated)", answer)

    def test_all_results_are_reachable(self) -> None:
        item = {"source": "test", "text": "test"}
        results = [(1.0, 1.0, item) for _ in range(18)]
        batches = split_into_batches(results, batch_size=5)
        self.assertEqual([len(batch) for batch in batches], [5, 5, 5, 3])

    def test_first_batch_contains_different_sources(self) -> None:
        results = [
            (1.0, 1.0, {"source": f"A {index}", "provider": "A", "text": "A"})
            for index in range(5)
        ]
        results.append(
            (0.5, 1.0, {"source": "B 1", "provider": "B", "text": "B"})
        )
        batches = split_into_source_batches(results, batch_size=5)
        providers = {item["provider"] for _, _, item in batches[0]}
        self.assertEqual(providers, {"A", "B"})
        self.assertEqual(sum(len(batch) for batch in batches), len(results))

if __name__ == "__main__":
    unittest.main()

# ChemRAG

ChemRAG is a local, evidence-grounded chemistry assistant for organic molecules
and inorganic materials. It combines live scientific retrieval, chemistry-aware
tools, local document search, and a Qwen model fine-tuned with LoRA.

This repository demonstrates a complete applied-AI workflow: source routing,
parallel retrieval, normalized scientific records, semantic reranking, local LLM
inference, LoRA fine-tuning, offline fallbacks, and automated evaluation.

## Capabilities

| Category            | Available questions                                                                                                                                                | Main evidence                                 |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------- |
| Organic compounds   | Formula, molecular weight, ALogP/XLogP, TPSA, hydrogen-bond counts, rotatable bonds, SMILES, InChI, and InChIKey                                                   | PubChem + ChEMBL                              |
| SMILES              | Validation, canonicalization, formula, molecular descriptors, and ring/bond information                                                                            | RDKit + PubChem                               |
| Inorganic materials | Crystal prototype, space group, lattice parameters, cell volume, band gap, formation energy, convex-hull distance, magnetism, and layer information when available | OQMD + COD + C2DB                             |
| SLICES              | Decode a SLICES representation, reconstruct its crystal information, and enrich it with material records                                                           | Official SLICES converter + inorganic sources |
| Local references    | Natural-language questions about local PDF, TXT, and Markdown files                                                                                                | FAISS document index                          |

Questions can be short or detailed and may be written in English or Spanish.
ChemRAG reports only retrieved evidence: if a requested value is unavailable, it
says so instead of inventing one.

### Example questions

```text
What is the molecular weight and canonical SMILES of aspirin?
What molecular properties are available for caffeine?
cid: 2244
inchikey: BSYNRYMUTXBXSQ-UHFFFAOYSA-N

smiles: CC(=O)OC1=CC=CC=C1C(=O)O
Validate this SMILES and report its formula, molecular weight, and TPSA:
CC(=O)OC1=CC=CC=C1C(=O)O

formula: BaTiO3
What band gap, formation energy, thermodynamic stability, and space group are
available for BaTiO3?
For MoS2, report the available electronic and structural properties.

slices: Nd Nd Si Si Ru Ru 0 3 -+o ...
Decode this SLICES string and report the reconstructed crystal information.

What does the included OQMD paper say about the size of the database?
```

Explicit prefixes are useful when an identifier could be ambiguous:
`name:`, `cid:`, `inchikey:`, `smiles:`, `formula:`, `material:`, and `slices:`.

ChemRAG is not an unrestricted general-knowledge chatbot. A question without a
recognized chemical entity is answered only when relevant evidence exists in the
local document index.

## How it works

```text
Question
   -> query router
      -> PubChem + ChEMBL                    organic compound
      -> OQMD + COD + C2DB                  inorganic formula
      -> RDKit + PubChem                    SMILES
      -> SLICES converter + material search SLICES
      -> FAISS                              local documents
   -> cross-encoder reranker
   -> complete reranked evidence context
   -> Qwen3-0.6B + chemistry LoRA
   -> evidence-backed scope validation for structured property lists
   -> final answer
```

Compatible web sources are queried concurrently. For an inorganic formula, all
three web integrations are attempted; the supplied local OQMD dataset is used
only when no web source returns a record. A failure from one provider is not
shown to the user when another provider can answer.

Online scientific records are normalized into one internal schema and labeled as
calculated values, reported values, or reported crystal data. After reranking,
every relevant record is included in the generation context. The same rule
applies to local documents, so a lower-ranked result is not silently discarded
when it contains a requested property.

FAISS does not permanently sort the chunks. It stores vectors and ranks them for
each new question according to vector similarity. A cross-encoder performs a
second, more precise ranking before generation.

The answer is produced in one model pass. The LoRA adapter is required and stays
active at its trained scale; the chatbot does not silently fall back to the base
Qwen model. For multi-property lists, a small deterministic validator reads the
normalized context, keeps exactly the requested fields, preserves evidence-type
labels, and marks missing fields as unavailable. Open-ended explanations remain
model-generated.

## Scientific sources and local tools

- **PubChem** and **ChEMBL** for organic molecules
- **OQMD**, **COD**, and **C2DB** for inorganic materials
- **RDKit** for local SMILES validation and molecular descriptors
- **SLICES** official converter for crystal reconstruction
- **FAISS** with multilingual E5 embeddings for local documents
- **Cross-encoder reranking** for evidence ordering
- **SQLite** fallback built from 561,888 supplied OQMD records

No database server or API key is required. An optional `HF_TOKEN` only improves
Hugging Face download limits. Retrieved records are cached in memory for the
current session; the cache does not grow permanently on disk.

> **Hardware requirement:** Running the complete project, including LoRA
> training, requires an NVIDIA GPU with CUDA support.

## Installation and first run

ChemRAG targets Python 3.11 on Windows and does not require a virtual
environment. Install a CUDA-compatible PyTorch build using the
[official PyTorch selector](https://pytorch.org/get-started/locally/), then run:

```powershell
py -m pip install -r requirements.txt
py src/build_oqmd_index.py
py src/create_embeddings.py
```

The LoRA artifact is loaded automatically from
`models/qwen3-0.6b-chemistry-lora`. If that directory is not present on a new
computer, create the training dataset and adapter once:

```powershell
py training/build_dataset.py --target-examples 10000
py training/check_training_setup.py
py training/train_lora.py
```

Start the assistant:

```powershell
py src/chat.py
```

Use `exit`, `quit`, or `salir` to close it. The first question is slower because
the embedding model, reranker, Qwen, and LoRA adapter must be loaded. They remain
in memory for later questions in the same session.

## Adding local knowledge

Place `.pdf`, `.txt`, or `.md` files in `data/knowledge`, then rebuild the FAISS
index:

```powershell
py src/create_embeddings.py
```

The supplied OQMD CSV and license are located in
`data/sources/oqmd_cgnn`. Rebuild its SQLite fallback with:

```powershell
py src/build_oqmd_index.py
```

Generated indexes are written to `data/processed` and are not committed.

## Fine-tuning

The base model is `Qwen/Qwen3-0.6B`. The dataset builder creates 10,000 balanced
organic and inorganic instruction examples by default, then makes a 90/10
entity-separated training and validation split. Fine-tuning teaches behavior
rather than current facts: selecting requested fields, preserving units and
method types, handling partial evidence, and refusing unsupported claims. RAG
supplies the current scientific data.

The adapter uses rank 16 and alpha 32. It targets every attention projection
(`q_proj`, `k_proj`, `v_proj`, and `o_proj`) and the feed-forward projections
(`gate_proj`, `up_proj`, and `down_proj`). Training is configured for one
complete pass over the training split, with validation and checkpoint selection
at the end of the epoch. The saved adapter is loaded at its normal LoRA scale
during chatbot inference.

## Evaluation

```powershell
py -m unittest discover -s tests -v
py evals/evaluate_retrieval.py
py evals/evaluate_batching.py
py evals/evaluate_generation.py
py evals/evaluate_finetuning.py
```

The first three evaluations validate routing, fallbacks, chemistry tools,
retrieval, complete evidence assembly, and grounded end-to-end answers.

`evaluate_finetuning.py` is an optional offline benchmark. It temporarily
disables LoRA only to compare the original Qwen baseline against Qwen + LoRA.
This comparison never runs inside `chat.py` and never affects production
answers; its purpose is to measure whether fine-tuning improves behavior rather
than merely assuming that it does.

## Project structure

```text
src/chat.py                 Terminal chatbot
src/search_chunks.py        Retrieval, context assembly, generation, and validation
src/chemistry/              Scientific APIs, schema, RDKit, and SLICES tools
src/local_llm.py            Qwen and LoRA loading/inference
src/create_embeddings.py    Local document indexing
src/build_oqmd_index.py     Local OQMD fallback indexing
training/                   Dataset generation and LoRA training
evals/                      Retrieval, generation, batching, and LoRA benchmarks
tests/                      Fast regression tests
```

Scientific values remain subject to their original provider licenses. The
supplied OQMD dataset includes its CC BY 4.0 attribution file.

import json

import faiss
import numpy as np

from load_document import (
    KNOWLEDGE_DIR,
    PROJECT_ROOT,
    find_documents,
    load_document,
    split_into_chunks,
)
from local_embeddings import (
    EMBEDDING_DIMENSION,
    create_document_embeddings,
)


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FAISS_INDEX_PATH = PROCESSED_DIR / "index.faiss"
METADATA_PATH = PROCESSED_DIR / "metadata.json"


def main() -> None:
    document_paths = find_documents(KNOWLEDGE_DIR)

    if not document_paths:
        raise FileNotFoundError(
            f"No reference documents were found in {KNOWLEDGE_DIR}. "
            "Online chemical records do not need to be pre-indexed."
        )

    records: list[dict[str, str]] = []

    for document_path in document_paths:
        document = load_document(document_path)

        for chunk in split_into_chunks(document):
            records.append(
                {
                    "source": str(
                        document_path.relative_to(KNOWLEDGE_DIR)
                    ),
                    "text": chunk,
                }
            )

    vectors = np.asarray(
        create_document_embeddings(
            [record["text"] for record in records]
        ),
        dtype=np.float32,
    )

    if vectors.ndim != 2:
        raise ValueError("Embeddings must form a matrix.")

    if vectors.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Unexpected dimension: {vectors.shape[1]}; "
            f"expected {EMBEDDING_DIMENSION}."
        )

    if len(records) != len(vectors):
        raise ValueError(
            "The number of chunks does not match the embeddings."
        )

    index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
    index.add(vectors)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    METADATA_PATH.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Reference documents: {len(document_paths)}")
    print(f"Indexed chunks: {index.ntotal}")
    print(f"Embedding dimensions: {index.d}")
    print(f"FAISS index: {FAISS_INDEX_PATH}")
    print(f"Metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()

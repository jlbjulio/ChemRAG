from pathlib import Path

from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = PROJECT_ROOT / "data" / "knowledge"

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def find_documents(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_document(path: Path) -> str:
    extension = path.suffix.lower()

    if extension in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")

    if extension == ".pdf":
        reader = PdfReader(path)

        return "\n\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    raise ValueError(
        f"Unsupported file format: {extension}"
    )


def split_into_chunks(
    text: str,
    max_words: int = 200,
    overlap_words: int = 30,
) -> list[str]:
    if overlap_words >= max_words:
        raise ValueError(
            "overlap_words must be smaller than max_words."
        )

    sections = [
        section.strip()
        for section in text.split("\n\n")
        if section.strip()
    ]

    chunks = []

    for section in sections:
        words = section.split()

        if len(words) <= max_words:
            chunks.append(section)
            continue

        step = max_words - overlap_words

        for start in range(0, len(words), step):
            chunk_words = words[start:start + max_words]

            if chunk_words:
                chunks.append(" ".join(chunk_words))

    return chunks


if __name__ == "__main__":
    documents = find_documents(KNOWLEDGE_DIR)

    print(f"Documents found: {len(documents)}")

    for document_path in documents:
        text = load_document(document_path)
        chunks = split_into_chunks(text)

        print(
            f"- {document_path.name}: "
            f"{len(chunks)} chunks"
        )

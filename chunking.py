"""Paragraph-aware chunking with overlap. Pure Python, no Azure needed, easy to test."""
import re


def _split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split one very long paragraph into windows of max_chars with overlap."""
    step = max(1, max_chars - overlap)
    return [text[i : i + max_chars] for i in range(0, len(text), step)]


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> list[str]:
    """Split text into chunks of roughly max_chars (a chunk can be up to about max_chars + overlap),
    keeping paragraphs together where possible. Each new chunk starts with the last `overlap`
    characters of the previous one so that sentences at the boundary are not lost."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[str] = []
    for p in paragraphs:
        pieces.extend(_split_long(p, max_chars, overlap) if len(p) > max_chars else [p])

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) + 2 > max_chars:
            chunks.append(current)
            current = current[-overlap:] + "\n\n" + piece if overlap else piece
        else:
            current = f"{current}\n\n{piece}" if current else piece
    if current.strip():
        chunks.append(current)
    return chunks


if __name__ == "__main__":
    sample = ("Paragraph one. " * 50 + "\n\n" + "Paragraph two. " * 50 + "\n\n" + "Short one.")
    for i, c in enumerate(chunk_text(sample, max_chars=300, overlap=50)):
        print(i, len(c), repr(c[:40]))

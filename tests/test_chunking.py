from rag.chunking import chunk_text


def test_chunk_text_splits_at_legal_headings() -> None:
    text = (
        "Article 1 Dispositions generales\n"
        "Ces clauses definissent le perimetre du contrat et la responsabilite des parties.\n\n"
        "Article 2 Tarifs et paiements\n"
        "Le client regle les honoraires sous trente jours et toute penalite est due apres mise en demeure."
    )

    chunks = chunk_text(text, chunk_size=120, overlap=0, use_tiktoken=False)

    assert any(chunk.startswith("Article 1") for chunk in chunks)
    assert any(chunk.startswith("Article 2") for chunk in chunks)


def test_chunk_text_returns_empty_for_blank_input() -> None:
    assert chunk_text("", chunk_size=50, overlap=0, use_tiktoken=False) == []

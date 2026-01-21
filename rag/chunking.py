from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(
    text: str,
    *,
    chunk_size: int,
    overlap: int,
    use_tiktoken: bool = True,
) -> list[str]:
    text = text.strip()
    if not text:
        return []
    
    # Define legal separators for French legal documents to preserve structure during chunking
    legal_separators = [
        r"\n(?=Article\s+(?:L\.)?\d+(?:[._-]\d+)?)",
        r"\n(?=Titre\s+(?:I{1,3}|IV|V|VI|VII|VIII|IX|\d+))",
        r"\n(?=Chapitre\s+(?:I{1,3}|IV|V|VI|VII|VIII|IX|\d+))",
        r"\n(?=(?:SECTION|Section)\s+(?:I{1,3}|IV|V|VI|VII|VIII|IX|\d+))",
        r"\n(?=§\s*\d+)",
        "\n\n",
        "\n",
        " ",
        "",
    ]

    if use_tiktoken:
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=legal_separators,
            is_separator_regex=True,
            keep_separator="start", # to keep headings with the chunk
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=legal_separators,
            is_separator_regex=True,
            keep_separator="start",
        )
    return splitter.split_text(text)

from pathlib import Path

from rag.preprocessing import preprocess_file


def test_preprocess_txt_compacts_lines(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    path.write_text("  First line  \n\nSecond line\n   \n", encoding="utf-8")

    result = preprocess_file(path)

    assert result == "First line\nSecond line"


def test_preprocess_csv_labels_and_skips_empty(tmp_path: Path) -> None:
    path = tmp_path / "table.csv"
    path.write_text(
        "name,amount,status\nAlice,100,paid\nBob,,pending\n,200,\n", encoding="utf-8"
    )

    result = preprocess_file(path).splitlines()

    assert "name: Alice | amount: 100 | status: paid" in result
    assert "name: Bob | status: pending" in result
    # The row with empty name/status should only include the non-empty value.
    assert "amount: 200" in result[-1]
    assert "name:" not in result[-1] and "status:" not in result[-1]


def test_preprocess_html_drops_noise_and_keeps_body(tmp_path: Path) -> None:
    path = tmp_path / "page.html"
    path.write_text(
        """
        <html>
            <head><script>ignore()</script></head>
            <body>
                <nav>menu</nav>
                <p>Main content stays.</p>
                <div id="cookie-banner">cookie notice</div>
                <footer>foot</footer>
            </body>
        </html>
        """,
        encoding="utf-8",
    )

    result = preprocess_file(path)

    assert "Main content stays." in result
    assert "menu" not in result
    assert "cookie notice" not in result

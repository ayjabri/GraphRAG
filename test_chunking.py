"""Tests for chunk_text() / chunk_documents(): boundaries, overlap, metadata, deterministic IDs.

Run with ``python test_chunking.py`` or ``pytest``.
"""

import main
from main import DOCS, EDGES, TraditionalRAG

chunk_text = TraditionalRAG.chunk_text
count = lambda text: len(TraditionalRAG.tokenize(text))  # noqa: E731 - index tokens


def spans(text, size, overlap):
    return [text[c["start"]:c["end"]] for c in chunk_text(text, size, overlap)]


def test_corpus_chunks_are_exact_traceable_spans():
    rag = TraditionalRAG(DOCS, EDGES, use_llm=False)
    for doc_id, doc in DOCS.items():
        ids = rag.doc_chunks[doc_id]
        assert len(ids) >= 2, f"{doc_id} should produce several chunks"
        assert ids == [f"{doc_id}#c{i}" for i in range(len(ids))]
        for chunk_id in ids:
            chunk = rag.chunks[chunk_id]
            assert chunk["text"] == doc["text"][chunk["start_char"]:chunk["end_char"]]  # traceable to the source
            assert chunk["doc_id"] == doc_id and chunk["acl"] == doc["acl"]           # ACL inherited
            assert chunk["n_chunks"] == len(ids) and chunk["title"] == doc["title"]
            assert chunk["token_count"] == count(chunk["text"]) <= main.CHUNKING["chunk_size"]


def test_chunk_ids_are_deterministic_and_local_to_their_document():
    first = TraditionalRAG(DOCS, EDGES, use_llm=False).chunks
    second = TraditionalRAG(DOCS, EDGES, use_llm=False).chunks
    assert first == second

    edited = {d: dict(v) for d, v in DOCS.items()}
    edited["general_ledger"]["text"] += "\n\nA new closing paragraph about ledger archiving."
    changed = TraditionalRAG(edited, EDGES, use_llm=False).chunks
    for chunk_id, chunk in first.items():  # other documents keep identical chunk IDs and text
        if chunk["doc_id"] != "general_ledger":
            assert changed[chunk_id]["text"] == chunk["text"]


def test_sentences_are_never_split():
    for doc in DOCS.values():
        text = doc["text"]
        for c in chunk_text(text, 400, 60):
            assert c["start"] == 0 or text[c["start"] - 1].isspace()          # starts at a unit boundary
            assert c["end"] == len(text) or text[c["end"]].isspace()           # ends at a unit boundary
            assert text[c["end"] - 1] in ".!?|:)" or text[c["end"] - 1].isalnum()


def test_overlap_repeats_whole_sentences_and_always_advances():
    text = " ".join(f"Sentence number {i} is here." for i in range(1, 21))  # 5 tokens each
    chunks = chunk_text(text, chunk_size=20, overlap=6)
    pieces = [text[c["start"]:c["end"]] for c in chunks]
    assert all(count(p) <= 20 for p in pieces)
    for previous, current in zip(chunks, chunks[1:]):
        assert current["start"] < previous["end"]                  # overlapping
        assert current["end"] > previous["end"]                    # but always adds new text (no duplicates)
        shared = text[current["start"]:previous["end"]]
        assert count(shared) <= 6 and shared.endswith(".")         # whole sentences only
    assert pieces[0].startswith("Sentence number 1 ") and pieces[-1].endswith("number 20 is here.")


def test_no_overlap_and_new_chunk_at_section_breaks():
    section = " ".join(f"Fact {i} about the ledger." for i in range(1, 7))  # 30 tokens
    text = f"# One\n\n{section}\n\n# Two\n\n{section.replace('ledger', 'payments')}"
    pieces = spans(text, size=60, overlap=10)
    assert pieces[0].startswith("# One") and pieces[0].endswith("ledger.")
    assert pieces[1].startswith("# Two") and "ledger" not in pieces[1]   # nothing carried across the heading


def test_small_sections_are_merged():
    text = "# A\n\nShort one.\n\n# B\n\nShort two.\n\n# C\n\nShort three."
    assert spans(text, size=40, overlap=5) == [text]


def test_lists_and_tables_are_units():
    text = "Intro sentence here.\n\n- first item\n- second item\n\n| a | b |\n| --- | --- |\n| 1 | 2 |"
    pieces = spans(text, size=3, overlap=0)  # each line is a unit; lines are never cut
    assert pieces == ["Intro sentence here.", "- first item", "- second item",
                      "| a | b |\n| --- | --- |", "| 1 | 2 |"]


def test_oversized_sentence_is_split_into_word_windows():
    long_sentence = " ".join(f"w{i}" for i in range(50)) + "."
    pieces = spans(long_sentence, size=20, overlap=5)
    assert len(pieces) == 3 and all(count(p) <= 20 for p in pieces)
    assert pieces[0].startswith("w0") and pieces[-1].endswith("w49.")


def test_document_strategy_and_bad_settings():
    whole = TraditionalRAG(DOCS, EDGES, use_llm=False, chunking={"strategy": "document"})
    assert len(whole.chunks) == len(DOCS)
    assert all(whole.chunks[f"{d}#c0"]["text"] == DOCS[d]["text"].rstrip() for d in DOCS)
    for bad in ({"chunk_size": 0}, {"chunk_size": 50, "overlap": 50}, {"strategy": "tokens"}):
        try:
            TraditionalRAG(DOCS, EDGES, use_llm=False, chunking=bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


def test_corpus_loader_validates_labels():
    corpus = main.load_corpus()
    assert set(corpus["docs"]) == set(DOCS) and corpus["edges"] == EDGES
    assert corpus["chunking"] == {"strategy": "sentence", "chunk_size": 400, "overlap": 60}
    for item in corpus["evals"]:
        assert item["id"] and item["groups"] and item["relevant"] and item["reference"]


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"OK: {len(tests)} chunking tests passed")

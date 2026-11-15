"""Run with:  python test_chunker.py   (no pytest needed)"""
from chunker import chunk_page, build_chunks, MAX_CHUNK

META = {"title": "Rules"}
BODY = """# Rules

Intro sentence for the page, long enough to matter.

## Tenants

""" + "\n".join(f"- item {i} " + "x" * 80 for i in range(60)) + """

## Quota

Short section about the quota.
"""


def test_heading_paths_and_size():
    chunks = chunk_page(META, BODY)
    paths = {p for p, _ in chunks}
    assert "Tenants" in paths and "Quota" in paths           # page H1 is not repeated in the path
    assert all(len(t) <= MAX_CHUNK for _, t in chunks)       # long list was split
    assert sum(1 for p, _ in chunks if p == "Tenants") > 1


def test_table_split_keeps_header():
    rows = "\n".join(f"| town{i} | {i} |" for i in range(300))
    body = f"## Table\n\n| Town | Rent |\n| --- | --- |\n{rows}\n"
    chunks = chunk_page(META, body)
    assert len(chunks) > 1
    assert all(t.startswith("| Town | Rent |") for _, t in chunks)


def test_ids_are_stable_and_unique():
    a, _ = build_chunks()
    b, _ = build_chunks()
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert len({c.uuid for c in a}) == len(a)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

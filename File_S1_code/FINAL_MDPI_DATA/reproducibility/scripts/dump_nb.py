import json, sys, pathlib
src = pathlib.Path(sys.argv[1]); out = pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
for nb in sorted(src.glob("*.ipynb")):
    d = json.loads(nb.read_text(encoding="utf-8"))
    lines = []
    for i, c in enumerate(d.get("cells", [])):
        if c.get("cell_type") != "code":
            continue
        code = "".join(c.get("source", []))
        lines.append(f"# ===== CELL {i} (exec {c.get('execution_count')}) =====\n{code}\n")
        # short text outputs
        for o in c.get("outputs", []):
            t = o.get("text") or (o.get("data", {}) or {}).get("text/plain")
            if t:
                t = "".join(t) if isinstance(t, list) else t
                t = t[:3000]
                lines.append("# --- OUTPUT ---\n" + "\n".join("#> " + l for l in t.splitlines()) + "\n")
    (out / (nb.stem + ".py.txt")).write_text("\n".join(lines), encoding="utf-8")
    print(nb.name, len(d.get("cells", [])))

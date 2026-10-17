"""Re-render F0B and F0C from the ORIGINAL 04_code/generate_F0_research_story.py with a smaller canvas so that the
same font sizes print larger in the journal layout. Content and data are unchanged (no new numbers).
Usage: python regen_F0B_F0C_larger_text.py <PROJECT_ROOT> <OUT_DIR>"""
import sys, types
from pathlib import Path

root, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
src = (root / "04_code" / "generate_F0_research_story.py").read_text(encoding="utf-8")
src = src.replace("from IPython.display import display", "display = lambda *a, **k: None")
src = src.replace("OUT=ROOT/'05_results'/'manuscript_final_L_to_O'/'figures'/'journal_ready'", f"OUT=Path(r'{out}')")
src = src.replace("fig,ax=plt.subplots(figsize=(13.8,7.4))", "fig,ax=plt.subplots(figsize=(10.2,6.6))")
src = src.replace("fig,axes=plt.subplots(1,3,figsize=(14.3,5.3),sharey=True)", "fig,axes=plt.subplots(1,3,figsize=(11.0,4.9),sharey=True)")
assert "figsize=(10.2,6.6)" in src and "figsize=(11.0,4.9)" in src
exec(compile(src, "generate_F0_research_story.py (re-rendered)", "exec"), {"__name__": "__main__", "__file__": str(root / "04_code" / "generate_F0_research_story.py")})

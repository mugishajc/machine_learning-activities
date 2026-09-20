"""Assemble the assignment into a single executable notebook."""
import re, nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

REPORT = "report/MIT91207_Assignment1_Mugisha_Jean_Claude_26016815.md"
SCRIPTS = {1: [], 2: ["q2_recommender_prep", "q2b_impact"], 3: ["q3_missing_data"],
           4: ["q4_encoding"], 5: ["q5_readmission", "q5b_leakage_scaling"],
           6: ["q6_crime"], 7: ["q7_regression", "q7b_complexity", "q7c_diagnostics"],
           8: ["q8_survey_eda"]}

raw = open(REPORT, encoding="utf8").read()

def strip_python_blocks(md):
    """Remove ```python excerpts: the live cell below shows the same code."""
    return re.sub(r"```python\n.*?\n```\n?", "", md, flags=re.S)

def to_inline(src):
    """Render figures in the notebook instead of writing them silently to disk."""
    src = src.replace('import matplotlib\nmatplotlib.use("Agg")\n', "import matplotlib\n")
    src = src.replace('matplotlib.use("Agg"); ', "")
    src = src.replace('matplotlib.use("Agg")\n', "")
    src = re.sub(r"plt\.close\(fig\)", "plt.show()", src)
    return src

cells = []
head, *rest = re.split(r"\n# Question ", raw)
appendix = ""
if "\n# Appendix" in rest[-1]:
    rest[-1], appendix = rest[-1].split("\n# Appendix", 1)

cells.append(new_markdown_cell(head.strip()))
cells.append(new_code_cell(
    "# Environment check. Every result below is produced by this notebook.\n"
    "%matplotlib inline\n"
    "import sys, pandas, numpy, sklearn, statsmodels, scipy, matplotlib\n"
    "print('python      ', sys.version.split()[0])\n"
    "for m in (pandas, numpy, sklearn, statsmodels, scipy, matplotlib):\n"
    "    print(f'{m.__name__:12s}', m.__version__)\n"
    "import os; print('\\nproject folder:', os.path.basename(os.getcwd()))\n"
    "print('datasets present:', sorted(os.listdir('data')))"))

for chunk in rest:
    qno = int(chunk[0])
    body = "# Question " + chunk.strip()
    cells.append(new_markdown_cell(strip_python_blocks(body).strip()))
    for s in SCRIPTS[qno]:
        src = open(f"code/{s}.py", encoding="utf8").read()
        cells.append(new_markdown_cell(f"### Question {qno} code: `code/{s}.py`"))
        cells.append(new_code_cell(to_inline(src).strip()))

if appendix:
    cells.append(new_markdown_cell("# Appendix" + appendix.strip()))

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python (MIT91207)", "language": "python", "name": "mluok"},
    "language_info": {"name": "python"}})
out = "MIT91207_Assignment1_Mugisha_Jean_Claude_26016815.ipynb"
nbf.write(nb, out)
print(f"wrote {out}: {len(cells)} cells "
      f"({sum(1 for c in cells if c.cell_type=='code')} code, "
      f"{sum(1 for c in cells if c.cell_type=='markdown')} markdown)")

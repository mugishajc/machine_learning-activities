"""Assemble the answers and analysis into one executable notebook."""
import re, nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

REPORT = "report/MIT91207_Assignment2_Mugisha_Jean_Claude_26016815.md"
SCRIPTS = {"Question Four": "q4_fraud_workflow",
           "Question Five": "q5_preprocessing_pipeline",
           "Question Six":  "q6_threshold_metrics",
           "Question Seven":"q7_regression_comparison",
           "Question Eight":"q8_kmeans"}
raw = open(REPORT, encoding="utf8").read()
cells = []
head, *rest = re.split(r"\n# (?=Question )", raw)
cells.append(new_markdown_cell(head.strip()))
cells.append(new_code_cell(
    "%matplotlib inline\n"
    "import sys, os, pandas, numpy, sklearn, scipy, matplotlib\n"
    "print('python      ', sys.version.split()[0])\n"
    "for m in (pandas, numpy, sklearn, scipy, matplotlib):\n"
    "    print(f'{m.__name__:12s}', m.__version__)\n"
    "print('\\ndatasets:', sorted(f for f in os.listdir('data') if f.endswith('.csv')))\n"
    "\n"
    "import warnings\n"
    "from sklearn.exceptions import ConvergenceWarning\n"
    "warnings.filterwarnings('ignore', category=ConvergenceWarning)\n"
    "warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')"))
for chunk in rest:
    body = "# " + chunk.strip()
    title = body.split("\n")[0].replace("# ", "").strip()
    for p in re.split(r"\n(?=## \d\([a-e]\))", body):
        if p.strip():
            cells.append(new_markdown_cell(re.sub(r"```python\n.*?\n```\n?", "", p, flags=re.S).strip()))
    for key, script in SCRIPTS.items():
        if title.startswith(key):
            src = open(f"code/{script}.py", encoding="utf8").read()
            src = src.replace('import matplotlib\nmatplotlib.use("Agg"); ', "import matplotlib\n")
            src = src.replace('matplotlib.use("Agg"); ', "").replace('matplotlib.use("Agg")\n', "")
            src = re.sub(r"plt\.close\(fig\)", "plt.show()", src)
            cells.append(new_markdown_cell(f"### {key}: analysis code (`code/{script}.py`)"))
            cells.append(new_code_cell(src.strip()))
nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.9.6"}})
out = "MIT91207_Assignment2_Mugisha_Jean_Claude_26016815.ipynb"
nbf.write(nb, out)
print(f"wrote {out}: {len(cells)} cells "
      f"({sum(1 for c in cells if c.cell_type=='code')} code, "
      f"{sum(1 for c in cells if c.cell_type=='markdown')} markdown)")

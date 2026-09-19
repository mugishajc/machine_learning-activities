"""Execute the assignment notebook in place, reporting progress per cell."""
import sys, time, nbformat
from nbclient import NotebookClient

NB = "MIT91207_Assignment1_Mugisha_Jean_Claude_26016815.ipynb"
nb = nbformat.read(NB, as_version=4)
code_cells = [i for i, c in enumerate(nb.cells) if c.cell_type == "code"]
print(f"cells: {len(nb.cells)} total, {len(code_cells)} to execute", flush=True)

client = NotebookClient(nb, timeout=5400, kernel_name="mluok",
                        resources={"metadata": {"path": "."}}, allow_errors=False)
t0 = time.time()
with client.setup_kernel():
    for n, i in enumerate(code_cells, 1):
        t = time.time()
        client.execute_cell(nb.cells[i], i)
        print(f"  [{n}/{len(code_cells)}] cell {i:2d} done in {time.time()-t:6.1f}s "
              f"(elapsed {time.time()-t0:6.1f}s)", flush=True)
nbformat.write(nb, NB)
print(f"written. total {time.time()-t0:.1f}s", flush=True)

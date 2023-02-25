import numpy as np
from qiskit import qpy

with open("circuit.qpy", "rb") as qpy_file_read:
    qc_loaded = qpy.load(qpy_file_read)[0]

print(qc_loaded.parameters)
qc_loaded = qc_loaded.bind_parameters([0.1, 0.1])
#print(qc_loaded)
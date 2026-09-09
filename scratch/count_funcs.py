import ast
import os

math_dir = r"c:\Users\yucel.pehlevan\concrete-fhe-toolkit\src\concrete_fhe_toolkit\math"
files = ["basic.py", "binary_division.py", "bits.py", "combinatorics.py", "fixed_point.py", "number_theory.py", "special.py"]

total_functions = 0
for f in files:
    path = os.path.join(math_dir, f)
    with open(path, "r", encoding="utf-8") as file:
        tree = ast.parse(file.read())
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and not node.name.startswith('_')]
        print(f"{f}: {len(functions)} functions -> {functions}")
        total_functions += len(functions)

print(f"Total public functions: {total_functions}")

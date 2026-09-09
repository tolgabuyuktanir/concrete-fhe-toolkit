import json
import os
import inspect
import importlib
from fractions import Fraction
import math

def create_cell(cell_type, source, outputs=None):
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" for line in source.split('\n')]
    }
    if cell["source"]:
        cell["source"][-1] = cell["source"][-1].rstrip('\n')
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = outputs or []
    return cell

def save_notebook(filename, cells):
    notebook = {
        "cells": cells,
        "metadata": {"language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5
    }
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

def generate_tests_for_module(mod_name, notebook_path):
    mod = importlib.import_module(f"concrete_fhe_toolkit.math.{mod_name}")
    functions = [name for name, obj in inspect.getmembers(mod, inspect.isfunction) if not name.startswith('_')]
    
    cells = []
    
    for fn_name in functions:
        func = getattr(mod, fn_name)
        sig = inspect.signature(func)
        
        desc = f"Tests the `{fn_name}` function. Validates correctness with various inputs including positives, negatives, and zeros."
        cells.append(create_cell("markdown", f"### Testing `{fn_name}`\n\n{desc}"))
        
        imports = f"from concrete import fhe\nfrom concrete_fhe_toolkit.math.{mod_name} import {fn_name}\n"
        if mod_name == 'combinatorics' or mod_name == 'number_theory' or mod_name == 'special':
            imports += "import math\n"
        
        test_code = ""
        # We will dispatch based on the name prefix
        
        try:
            if fn_name.startswith("compile_"):
                # It returns a circuit. We need to call it with appropriate bounds.
                kwargs = {}
                for param_name, param in sig.parameters.items():
                    if param_name in ('min_left', 'min_right', 'min_numerator', 'min_denominator', 'min_value', 'min_input'):
                        kwargs[param_name] = -3
                    elif param_name in ('max_left', 'max_right', 'max_numerator', 'max_denominator', 'max_value', 'max_input'):
                        kwargs[param_name] = 3
                    elif param_name == 'size': kwargs['param_name'] = 3
                    elif param_name == 'multiplier': kwargs['param_name'] = 2
                    elif param_name == 'absolute_tolerance': kwargs['param_name'] = 1
                    elif param_name == 'zero_result': kwargs['param_name'] = 7
                    elif param_name == 'zero_quotient': kwargs['param_name'] = 7
                    elif param_name == 'zero_remainder': kwargs['param_name'] = 7
                    elif param_name == 'scale': kwargs['param_name'] = 10
                    elif param_name == 'input_scale': kwargs['param_name'] = 10
                    elif param_name == 'output_scale': kwargs['param_name'] = 10
                    elif param_name == 'angle_unit': kwargs['param_name'] = "degrees"
                    elif param_name == 'configuration': pass
                    elif param.default != inspect.Parameter.empty:
                        kwargs[param_name] = param.default
                    else:
                        # Fallback for some hardcoded ones
                        if param_name == 'k': kwargs['param_name'] = 2
                        elif param_name == 'numerator_width': kwargs['param_name'] = 4
                        elif param_name == 'denominator_width': kwargs['param_name'] = 3
                        elif param_name == 'fractional_bits': kwargs['param_name'] = 3
                        
                args_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
                test_code += f"circuit = {fn_name}({args_str})\n\n"
                
                # Now generate assertions based on simulate
                # This is tricky because we don't have the python equivalent easily accessible in the test string unless we write it out.
                # Actually, the user wants us to use `fhe.Compiler` and encrypt_run_decrypt.
                
                # ...
        except Exception as e:
            test_code += f"# Error generating test: {e}\n"
            
        # We need a robust way to generate these test strings!

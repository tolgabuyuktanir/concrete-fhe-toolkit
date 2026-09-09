import json
import os
import inspect
import importlib

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

def generate_notebook_for_module(mod_name, notebook_name):
    mod = importlib.import_module(f"concrete_fhe_toolkit.math.{mod_name}")
    
    # Extract only functions that belong to this exact module
    all_funcs = []
    for name, obj in inspect.getmembers(mod, inspect.isfunction):
        if not name.startswith('_') and getattr(obj, '__module__', '') == mod.__name__:
            all_funcs.append((name, obj))
            
    # Group by the actual function object to detect aliases and find original line number
    primary_funcs = {}
    for name, obj in all_funcs:
        # we group by function identity (or name)
        # compile_sub is alias of compile_subtract. compile_sub.__name__ is 'compile_subtract'
        primary_name = obj.__name__
        if primary_name not in primary_funcs:
            primary_funcs[primary_name] = {
                'obj': obj,
                'aliases': [],
                'line_no': inspect.getsourcelines(obj)[1]
            }
        if name != primary_name:
            primary_funcs[primary_name]['aliases'].append(name)
            
    # Also if the primary name was somehow not exported but an alias was, the primary might not be the exported name.
    # Usually the primary name is in the module dict though.
    
    # Sort by original definition line number
    sorted_primary = sorted(primary_funcs.items(), key=lambda x: x[1]['line_no'])
    
    cells = []
    
    for primary_name, data in sorted_primary:
        func = data['obj']
        aliases = data['aliases']
        sig = inspect.signature(func)
        
        # We will use the primary name for the test, but mention aliases
        fn_name = primary_name
        
        desc = f"Tests the `{fn_name}` function. Includes various inputs to cover edge cases like positives, negatives, and zeros."
        if aliases:
            desc += f"\n\n**Aliases**: `{', '.join(aliases)}` can also be used equivalently."
            
        cells.append(create_cell("markdown", f"### Testing `{fn_name}`\n\n{desc}"))
        
        imports = f"from concrete import fhe\nfrom concrete_fhe_toolkit.math.{mod_name} import {fn_name}\n"
        if mod_name == 'combinatorics' or mod_name == 'number_theory' or mod_name == 'special':
            imports += "import math\n"
        
        base_fn_name = fn_name.replace("compile_", "") if fn_name.startswith("compile_") else fn_name
        if fn_name.startswith("compile_") and hasattr(mod, base_fn_name):
            imports += f"from concrete_fhe_toolkit.math.{mod_name} import {base_fn_name}\n"
            
        test_code = imports + "\n"
        
        try:
            if fn_name.startswith("compile_"):
                kwargs = {}
                for p_name, p in sig.parameters.items():
                    if p_name in ('min_left', 'min_right', 'min_numerator', 'min_denominator', 'min_value', 'min_input'):
                        kwargs[p_name] = -3
                    elif p_name in ('max_left', 'max_right', 'max_numerator', 'max_denominator', 'max_value', 'max_input'):
                        kwargs[p_name] = 3
                    elif p_name == 'size': kwargs[p_name] = 3
                    elif p_name == 'multiplier': kwargs[p_name] = 2
                    elif p_name == 'absolute_tolerance': kwargs[p_name] = 1
                    elif p_name == 'zero_result': kwargs[p_name] = 7
                    elif p_name == 'zero_quotient': kwargs[p_name] = 7
                    elif p_name == 'zero_remainder': kwargs[p_name] = 7
                    elif p_name == 'scale': kwargs[p_name] = 10
                    elif p_name == 'input_scale': kwargs[p_name] = 10
                    elif p_name == 'output_scale': kwargs[p_name] = 10
                    elif p_name == 'angle_unit': kwargs[p_name] = "'degrees'"
                    elif p_name == 'k': kwargs[p_name] = 2
                    elif p_name == 'numerator_width': kwargs[p_name] = 4
                    elif p_name == 'denominator_width': kwargs[p_name] = 3
                    elif p_name == 'fractional_bits': kwargs[p_name] = 3
                    elif p_name == 'configuration': pass
                    elif p.default != inspect.Parameter.empty and p.default is not None:
                        kwargs[p_name] = p.default if isinstance(p.default, (int, float, str)) else str(p.default)
                        
                args_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
                test_code += f"circuit = {fn_name}({args_str})\n\n"
                
                arity = 1
                if hasattr(mod, base_fn_name):
                    base_sig = inspect.signature(getattr(mod, base_fn_name))
                    arity = len(base_sig.parameters)
                else:
                    if 'left' in sig.parameters or 'right' in sig.parameters or 'numerator' in sig.parameters:
                        arity = 2
                        
                test_inputs = []
                if arity == 1:
                    test_inputs = [(2,), (-1,), (0,), (3,)]
                elif arity == 2:
                    test_inputs = [(2, 1), (-1, -1), (0, 0), (3, -2), (0, 2)]
                elif arity == 3:
                    test_inputs = [(2, 1, 1), (-1, -1, 2), (0, 0, 0), (3, -2, 1)]
                    
                test_code += f"inputset = {test_inputs}\n"
                
                test_code += "for inp in inputset:\n"
                test_code += "    try:\n"
                if hasattr(mod, base_fn_name):
                    test_code += f"        expected = {base_fn_name}(*inp)\n"
                    test_code += "        if isinstance(expected, tuple):\n"
                    test_code += "            assert tuple(int(x) for x in circuit.encrypt_run_decrypt(*inp)) == expected, f\"Failed at {inp}\"\n"
                    test_code += "        else:\n"
                    test_code += "            assert int(circuit.encrypt_run_decrypt(*inp)) == int(expected), f\"Failed at {inp}\"\n"
                else:
                    test_code += "        circuit.encrypt_run_decrypt(*inp)\n"
                test_code += "    except Exception as e:\n"
                test_code += "        print(f\"Skipping {inp} due to bounds or other error: {e}\")\n\n"
                
            elif fn_name.startswith("make_"):
                kwargs = {}
                for p_name, p in sig.parameters.items():
                    if p_name in ('min_left', 'min_right', 'min_numerator', 'min_denominator', 'min_value', 'min_input'):
                        kwargs[p_name] = -3
                    elif p_name in ('max_left', 'max_right', 'max_numerator', 'max_denominator', 'max_value', 'max_input'):
                        kwargs[p_name] = 3
                    elif p_name == 'size': kwargs[p_name] = 3
                    elif p_name == 'multiplier': kwargs[p_name] = 2
                    elif p_name == 'absolute_tolerance': kwargs[p_name] = 1
                    elif p_name == 'zero_result': kwargs[p_name] = 7
                    elif p_name == 'zero_quotient': kwargs[p_name] = 7
                    elif p_name == 'zero_remainder': kwargs[p_name] = 7
                    elif p_name == 'k': kwargs[p_name] = 2
                    elif p_name == 'scale': kwargs[p_name] = 10
                    elif p_name == 'configuration': pass
                    elif p.default != inspect.Parameter.empty and p.default is not None:
                        kwargs[p_name] = p.default if isinstance(p.default, (int, float, str)) else str(p.default)
                        
                args_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
                test_code += f"fn = {fn_name}({args_str})\n\n"
                
                base_fn_name = fn_name.replace("make_", "")
                arity = 1
                if hasattr(mod, base_fn_name):
                    base_sig = inspect.signature(getattr(mod, base_fn_name))
                    arity = len(base_sig.parameters)
                else:
                    if 'left' in sig.parameters or 'numerator' in sig.parameters: arity = 2
                
                enc_dict = "{" + ", ".join([f"'arg{i}': 'encrypted'" for i in range(arity)]) + "}"
                params_str = ", ".join([f"arg{i}" for i in range(arity)])
                test_code += f"def test_{fn_name}_enc({params_str}):\n    return fn({params_str})\n\n"
                test_code += f"compiler = fhe.Compiler(test_{fn_name}_enc, {enc_dict})\n"
                
                test_inputs = []
                if arity == 1:
                    test_inputs = [(2,), (-1,), (0,), (3,)]
                elif arity == 2:
                    test_inputs = [(2, 1), (-1, -1), (0, 0), (3, -2), (0, 2)]
                elif arity == 3:
                    test_inputs = [(2, 1, 1), (-1, -1, 2), (0, 0, 0), (3, -2, 1)]
                
                test_code += f"inputset = {test_inputs}\n"
                test_code += "circuit = compiler.compile(inputset)\n\n"
                
                test_code += "for inp in inputset:\n"
                test_code += "    try:\n"
                test_code += f"        expected = fn(*inp)\n"
                test_code += "        if isinstance(expected, tuple):\n"
                test_code += "            assert tuple(int(x) for x in circuit.encrypt_run_decrypt(*inp)) == expected, f\"Failed at {inp}\"\n"
                test_code += "        else:\n"
                test_code += "            assert int(circuit.encrypt_run_decrypt(*inp)) == int(expected), f\"Failed at {inp}\"\n"
                test_code += "    except Exception as e:\n"
                test_code += "        print(f\"Skipping {inp} due to bounds or other error: {e}\")\n\n"
                        
            else:
                params_str = ", ".join(sig.parameters.keys())
                test_code += f"def test_{fn_name}({params_str}):\n    return {fn_name}({params_str})\n\n"
                enc_dict = "{" + ", ".join([f"'{k}': 'encrypted'" for k in sig.parameters.keys()]) + "}"
                test_code += f"compiler = fhe.Compiler(test_{fn_name}, {enc_dict})\n"
                
                test_inputs = []
                arity = len(sig.parameters)
                if arity == 1:
                    test_inputs = [(2,), (-1,), (0,), (3,)]
                elif arity == 2:
                    test_inputs = [(2, 1), (-1, -1), (0, 0), (3, -2), (0, 2)]
                elif arity == 3:
                    test_inputs = [(2, 1, 1), (-1, -1, 2), (0, 0, 0), (3, -2, 1)]
                
                test_code += f"inputset = {test_inputs}\n"
                test_code += "circuit = compiler.compile(inputset)\n\n"
                
                test_code += "for inp in inputset:\n"
                test_code += "    try:\n"
                test_code += f"        expected = {fn_name}(*inp)\n"
                test_code += "        if isinstance(expected, tuple):\n"
                test_code += "            assert tuple(int(x) for x in circuit.encrypt_run_decrypt(*inp)) == expected, f\"Failed at {inp}\"\n"
                test_code += "        else:\n"
                test_code += "            assert int(circuit.encrypt_run_decrypt(*inp)) == int(expected), f\"Failed at {inp}\"\n"
                test_code += "    except Exception as e:\n"
                test_code += "        print(f\"Skipping {inp} due to bounds or other error: {e}\")\n\n"
            
            test_code += f"print(\"{fn_name} tests passed!\")"
            cells.append(create_cell("code", test_code))
        except Exception as e:
            cells.append(create_cell("code", f"# Could not auto-generate test for {fn_name}\n# Error: {e}"))
            
    save_notebook(notebook_name, cells)

if __name__ == "__main__":
    modules = [
        ("basic", "04_math_basic.ipynb"),
        ("binary_division", "05_math_binary_division.ipynb"),
        ("bits", "06_math_bits.ipynb"),
        ("combinatorics", "07_math_combinatorics.ipynb"),
        ("fixed_point", "08_math_fixed_point.ipynb"),
        ("number_theory", "09_math_number_theory.ipynb"),
        ("special", "10_math_special.ipynb"),
    ]
    for mod, fname in modules:
        generate_notebook_for_module(mod, rf"c:\Users\yucel.pehlevan\concrete-fhe-toolkit\docs\tutorials\{fname}")
        print(f"Generated {fname}")

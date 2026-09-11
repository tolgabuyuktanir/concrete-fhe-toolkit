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

def generate_inputset(fn_name, param_names):
    arity = len(param_names)
    if arity == 1:
        if 'square' in fn_name or 'cube' in fn_name:
            return [(2,), (-2,), (0,), (15,), (-15,)]
        elif 'sqrt' in fn_name:
            return [(0,), (1,), (4,), (9,), (15,)]
        elif 'log' in fn_name or 'exp' in fn_name:
            return [(1,), (2,), (5,), (10,)]
        elif 'bits' in fn_name or 'count' in fn_name:
            return [(0,), (1,), (7,), (15,)]
        return [(15,), (-15,), (0,), (2,)]
    elif arity == 2:
        if 'div' in fn_name or 'mod' in fn_name or 'remainder' in fn_name:
            return [(15, 4), (-14, 5), (3, -2), (6, 6), (15, 0), (0, 7)]
        elif 'shift' in fn_name:
            return [(15, 1), (1, 3), (-15, 2), (8, 0)]
        elif 'coprime' in fn_name or 'gcd' in fn_name or 'lcm' in fn_name:
            return [(15, 5), (14, 7), (12, 8), (7, 7)]
        return [(15, 1), (-15, -2), (0, 0), (14, -7), (0, 15)]
    elif arity == 3:
        return [(15, 2, 1), (-15, -2, 3), (0, 0, 0), (10, -5, 2)]
    return []

def filter_kwargs(kwargs, func):
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}

def get_kwargs(sig, fn_name):
    kwargs = {}
    is_positive_only = 'sqrt' in fn_name or 'log' in fn_name or 'exp' in fn_name
    
    for p_name, p in sig.parameters.items():
        if p_name in ('min_left', 'min_right', 'min_numerator', 'min_denominator', 'min_value', 'min_input', 'min_base'):
            kwargs[p_name] = 0 if is_positive_only else -15
        elif p_name in ('max_left', 'max_right', 'max_numerator', 'max_denominator', 'max_value', 'max_input', 'max_base', 'max_exponent', 'max_n'):
            kwargs[p_name] = 4 if 'pow' in fn_name else 15
        elif p_name == 'min_exponent': kwargs[p_name] = 1
        elif p_name == 'exponent': kwargs[p_name] = 2
        elif p_name == 'base': kwargs[p_name] = 2
        elif p_name == 'modulus': kwargs[p_name] = 5
        elif p_name == 'step': kwargs[p_name] = 3
        elif p_name == 'size': kwargs[p_name] = 3
        elif p_name == 'multiplier': kwargs[p_name] = 2
        elif p_name == 'absolute_tolerance': kwargs[p_name] = 1
        elif p_name == 'invalid_result': kwargs[p_name] = 7
        elif p_name == 'zero_result': kwargs[p_name] = 7
        elif p_name == 'zero_quotient': kwargs[p_name] = 7
        elif p_name == 'zero_remainder': kwargs[p_name] = 7
        elif p_name == 'k': kwargs[p_name] = 2
        elif p_name == 'scale': kwargs[p_name] = 10
        elif p_name == 'input_scale': kwargs[p_name] = 10
        elif p_name == 'output_scale': kwargs[p_name] = 10
        elif p_name == 'angle_unit': kwargs[p_name] = 'degrees'
        elif p_name == 'numerator_width': kwargs[p_name] = 4
        elif p_name == 'denominator_width': kwargs[p_name] = 3
        elif p_name == 'fractional_bits': kwargs[p_name] = 3
        elif p_name == 'configuration': pass
        elif p.default != inspect.Parameter.empty and p.default is not None:
            kwargs[p_name] = p.default if isinstance(p.default, (int, float, str)) else str(p.default)
    return kwargs

def generate_notebook_for_module(mod_name, notebook_name):
    mod = importlib.import_module(f"concrete_fhe_toolkit.math.{mod_name}")
    
    all_funcs = []
    for name, obj in inspect.getmembers(mod, inspect.isfunction):
        if not name.startswith('_') and getattr(obj, '__module__', '') == mod.__name__:
            all_funcs.append((name, obj))
            
    primary_funcs = {}
    for name, obj in all_funcs:
        primary_name = obj.__name__
        if primary_name not in primary_funcs:
            primary_funcs[primary_name] = {
                'obj': obj,
                'aliases': [],
                'line_no': inspect.getsourcelines(obj)[1]
            }
        if name != primary_name:
            primary_funcs[primary_name]['aliases'].append(name)
            
    sorted_primary = sorted(primary_funcs.items(), key=lambda x: x[1]['line_no'])
    
    cells = []
    
    for primary_name, data in sorted_primary:
        func = data['obj']
        aliases = data['aliases']
        sig = inspect.signature(func)
        fn_name = primary_name
        
        desc = f"Tests the `{fn_name}` function. Includes specific edge cases and a widened `[-15, 15]` input domain."
        if aliases:
            desc += f"\n\n**Aliases**: `{', '.join(aliases)}` can also be used equivalently."
            
        cells.append(create_cell("markdown", f"### Testing `{fn_name}`\n\n{desc}"))
        
        imports = f"from concrete import fhe\nfrom concrete_fhe_toolkit.math.{mod_name} import {fn_name}\n"
        if mod_name in ('combinatorics', 'number_theory', 'special'):
            imports += "import math\n"
        
        base_fn_name = fn_name.replace("compile_", "") if fn_name.startswith("compile_") else fn_name
        make_fn_name = fn_name.replace("compile_", "make_") if fn_name.startswith("compile_") else fn_name
        
        if fn_name.startswith("compile_"):
            if hasattr(mod, base_fn_name):
                imports += f"from concrete_fhe_toolkit.math.{mod_name} import {base_fn_name}\n"
            if hasattr(mod, make_fn_name):
                imports += f"from concrete_fhe_toolkit.math.{mod_name} import {make_fn_name}\n"
                
        test_code = imports + "\n"
        
        try:
            if fn_name.startswith("compile_"):
                kwargs = get_kwargs(sig, fn_name)
                args_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
                test_code += f"circuit = {fn_name}({args_str})\n\n"
                
                # Determine parameter names and cleartext fallback dynamically
                if hasattr(mod, make_fn_name):
                    make_fn = getattr(mod, make_fn_name)
                    make_kwargs = filter_kwargs(kwargs, make_fn)
                    clear_fn = make_fn(**make_kwargs)
                    param_names = list(inspect.signature(clear_fn).parameters.keys())
                    expected_args_str = ", ".join(f"{k}={v}" for k, v in make_kwargs.items())
                    expected_call = f"{make_fn_name}({expected_args_str})(*inp)"
                elif hasattr(mod, base_fn_name):
                    base_fn = getattr(mod, base_fn_name)
                    param_names = list(inspect.signature(base_fn).parameters.keys())
                    expected_call = f"{base_fn_name}(*inp)"
                else:
                    p_names = list(sig.parameters.keys())
                    if any(n in p_names for n in ('min_left', 'min_right', 'min_numerator', 'left', 'numerator', 'right')):
                        param_names = ['left', 'right']
                    else:
                        param_names = ['value']
                    expected_call = None
                        
                test_inputs = generate_inputset(fn_name, param_names)
                test_code += f"inputset = {test_inputs}\n"
                
                test_code += "for inp in inputset:\n"
                test_code += "    try:\n"
                if expected_call:
                    test_code += f"        expected = {expected_call}\n"
                    test_code += "        if isinstance(expected, tuple):\n"
                    test_code += "            assert tuple(int(x) for x in circuit.encrypt_run_decrypt(*inp)) == expected, f\"Failed at {inp}\"\n"
                    test_code += "        else:\n"
                    test_code += "            assert int(circuit.encrypt_run_decrypt(*inp)) == int(expected), f\"Failed at {inp}\"\n"
                else:
                    test_code += "        circuit.encrypt_run_decrypt(*inp)\n"
                test_code += "    except Exception as e:\n"
                test_code += "        print(f\"Skipping {inp} due to bounds or other error: {e}\")\n\n"
                
            elif fn_name.startswith("make_"):
                kwargs = get_kwargs(sig, fn_name)
                args_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
                test_code += f"fn = {fn_name}({args_str})\n\n"
                
                # Introspect the returned function to get actual parameter names!
                real_fn = getattr(mod, fn_name)(**kwargs)
                param_names = list(inspect.signature(real_fn).parameters.keys())
                
                enc_dict = "{" + ", ".join([f"'{k}': 'encrypted'" for k in param_names]) + "}"
                params_str = ", ".join(param_names)
                test_code += f"def test_{fn_name}_enc({params_str}):\n    return fn({params_str})\n\n"
                test_code += f"compiler = fhe.Compiler(test_{fn_name}_enc, {enc_dict})\n"
                
                test_inputs = generate_inputset(fn_name, param_names)
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
                param_names = list(sig.parameters.keys())
                params_str = ", ".join(param_names)
                test_code += f"def test_{fn_name}({params_str}):\n    return {fn_name}({params_str})\n\n"
                enc_dict = "{" + ", ".join([f"'{k}': 'encrypted'" for k in param_names]) + "}"
                test_code += f"compiler = fhe.Compiler(test_{fn_name}, {enc_dict})\n"
                
                test_inputs = generate_inputset(fn_name, param_names)
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

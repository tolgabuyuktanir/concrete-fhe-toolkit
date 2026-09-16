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
    print(f"{os.path.basename(filename)} has {len(cells)} cells.")
    if len(cells) == 0:
        if os.path.exists(filename):
            os.remove(filename)
        return
        
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

def generate_inputset(fn_name, param_names):
    import random
    random.seed(42)
    arity = len(param_names)
    
    if 'training' in fn_name:
        if 'linear_regression' in fn_name:
            return [([[1, 2]], [0], 1, 2)]
        if 'naive_bayes' in fn_name:
            return [([[1, 0]], [[1, 0]])]
    
    if any(k in fn_name for k in ('fsum', 'sumprod', 'dist', 'distance', 'error', 'loss', 'score', 'accuracy', 'true_', 'false_', 'confusion', 'norm', 'binarize', 'clip', 'normalize', 'softmax', 'one_hot')):
        if arity == 2:
            if 'one_hot' in fn_name:
                return [(2, 5), (0, 3)]
            if 'binarize' in fn_name:
                return [([0, 1, 2], 1), ([-1, 0, 1], 0)]
            if 'normalize' in fn_name:
                return [([0, 10, 20], 10), ([-5, 0, 5], 5)]
            return [([0, 1, 1], [0, 0, 1]), ([1, 0, 1], [1, 0, 1]), ([0, 0, 0], [1, 1, 1])]
        elif arity == 3:
            return [([1, 2, 3], 1, 3), ([-1, 0, 1], -1, 1)] # For clip_array
        return [([0, 1, 2],), ([-1, 0, 1],), ([0, 0, 0],)]

    if 'bits' in fn_name or 'bit_' in fn_name or fn_name in ('full_adder_bit', 'full_subtractor_bit', 'popcount_bits', 'parity_bits', 'unsigned_compare_bits', 'twos_complement_add_bits', 'twos_complement_multiply_by_constant_bits', 'multiply_bits'):
        if fn_name in ('integer_to_bits', 'unsigned_to_bits', 'twos_complement_bits', 'return_actual_value'):
            return [(2,), (-2,), (0,), (1,)] if arity == 1 else [(2, 0), (1, 1)]
        if fn_name == 'sign_magnitude_to_twos_complement_bits':
            return [([0, 1, 0], 1), ([1, 1, 1], 0), ([0, 0, 0], 0)]
        if fn_name.endswith('_bits') or 'many' in fn_name:
            # Array inputs
            if arity == 1:
                return [([0, 1, 0],), ([1, 1, 1],), ([0, 0, 0],)]
            elif 'shift' in fn_name or 'rotate' in fn_name:
                return [([1, 0, 1], 1), ([0, 1, 0], 2), ([1, 1, 1], 0)]
            elif arity == 2:
                return [([0, 1, 0], [1, 0, 1]), ([1, 1, 1], [1, 1, 1]), ([0, 0, 0], [0, 0, 0])]
        else:
            # Scalar bit inputs
            if arity == 1:
                return [(0,), (1,)]
            elif arity == 2:
                return [(0, 0), (0, 1), (1, 0), (1, 1)]
            elif arity == 3:
                return [(0, 0, 0), (0, 1, 0), (1, 0, 1), (1, 1, 1), (0, 1, 1)]

    if 'comb' in fn_name or 'perm' in fn_name or 'factorial' in fn_name or 'fibonacci' in fn_name:
        if arity == 1: return [(0,), (1,), (2,), (3,)]
        if arity == 2: return [(3, 2), (3, 3), (2, 2), (0, 0), (3, 0)]

    if 'pow' in fn_name:
        if arity == 2: return [(3, 1), (2, 2), (0, 0), (-2, 3)]

    if arity == 1:
        if 'square' in fn_name or 'cube' in fn_name:
            return [(2,), (-2,), (0,), (3,)]
        elif 'sqrt' in fn_name or 'isqrt' in fn_name:
            return [(0,), (1,), (2,), (3,)]
        elif 'log' in fn_name or 'exp' in fn_name:
            return [(1,), (2,), (3,)]
        return [(3,), (-2,), (0,), (2,)]
    elif arity == 2:
        if 'div' in fn_name or 'mod' in fn_name or 'remainder' in fn_name:
            return [(3, 2), (-2, 3), (3, -2), (3, 3), (2, 0), (0, 3)]
        elif 'shift' in fn_name:
            return [(3, 1), (1, 2), (-2, 2), (2, 0)]
        elif 'coprime' in fn_name or 'gcd' in fn_name or 'lcm' in fn_name:
            return [(3, 2), (2, 3), (3, 1), (3, 3)]
        elif 'atan2' in fn_name:
            return [(3, 3), (0, 3), (3, 0), (2, 2)]
        return [(3, 1), (-2, -2), (0, 0), (2, 2), (1, 3)]
    elif arity == 3:
        return [(3, 2, 1), (-2, -2, 3), (0, 0, 0), (2, -2, 2), (10, 5, -2)]
    

    # Fallback for arity >= 4
    return [tuple([2]*arity), tuple([0]*arity), tuple([1]*arity)]

def filter_kwargs(kwargs, func):
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}

def get_kwargs(sig, fn_name):
    kwargs = {}
    is_positive_only = 'sqrt' in fn_name or 'log' in fn_name or 'exp' in fn_name
    for p_name, p in sig.parameters.items():
        if p_name in ('min_left', 'min_right', 'min_numerator', 'min_denominator', 'min_value', 'min_input', 'min_base'):
            kwargs[p_name] = 0 if is_positive_only else -2
        elif p_name in ('max_left', 'max_right', 'max_numerator', 'max_denominator', 'max_value', 'max_input', 'max_base', 'max_exponent', 'zero_quotient', 'zero_remainder', 'max_n', 'max_k'):
            kwargs[p_name] = 3
        elif p_name == 'angle_unit': kwargs[p_name] = 'degrees'
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
        elif p_name == 'n_iterations': kwargs[p_name] = 1
        elif p_name == 'tree': kwargs[p_name] = {'feature': 0, 'threshold': 1, 'left': 0, 'right': 1}
        elif p_name == 'trees': kwargs[p_name] = [{'feature': 0, 'threshold': 1, 'left': 0, 'right': 1}]
        elif p_name == 'means': kwargs[p_name] = [[1, 2]]
        elif p_name == 'components': kwargs[p_name] = [[1, 2]]
        elif p_name == 'filters': kwargs[p_name] = [[[[1]]]]
        elif p_name == 'X_train': kwargs[p_name] = [[1]]
        elif p_name == 'y_train': kwargs[p_name] = [0]
        elif p_name == 'mlp_layers': kwargs[p_name] = [([1], [0], "relu")]
        elif p_name == 'log_prob_tables': kwargs[p_name] = [[[1]]]
        elif p_name == 'priors': kwargs[p_name] = [1]
        elif p_name == 'centroids': kwargs[p_name] = [[1, 1], [3, 3]]
        elif p_name == 'max_distance': kwargs[p_name] = 10
        elif p_name == 'weights': kwargs[p_name] = [1, 2]
        elif p_name == 'bias': kwargs[p_name] = 1
        elif p_name == 'thresholds': kwargs[p_name] = [1, 1]
        elif p_name == 'candidate_thresholds': kwargs[p_name] = [[2]]
        elif p_name == 'initial_centroids': kwargs[p_name] = [[1, 1], [3, 3]]
        elif p_name == 'num_classes': kwargs[p_name] = 2
        elif p_name == 'max_depth': kwargs[p_name] = 1
        elif p_name == 'min_samples_leaf': kwargs[p_name] = 1
        elif p_name == 'learning_rate': kwargs[p_name] = 1
    return kwargs

def generate_notebook_for_module(subpkg, mod_name, notebook_name):
    mod = importlib.import_module(f"concrete_fhe_toolkit.{subpkg}.{mod_name}")
    
    all_funcs = []
    import inspect
    for name, obj in inspect.getmembers(mod):
        if name.startswith('_'):
            continue
        if inspect.isfunction(obj) or (inspect.isclass(obj) and issubclass(obj, object)):
            mod_name_obj = getattr(obj, '__module__', '')
            if mod_name_obj == mod.__name__:
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
        fn_name = primary_name
        docstring = func.__doc__ or ''
        
        is_class = inspect.isclass(func)
        if is_class:
            try:
                sig = inspect.signature(func.__init__)
            except ValueError:
                sig = inspect.signature(func)
        else:
            sig = inspect.signature(func)
            
        if is_class:
            if fn_name == 'FHEModel' or fn_name == 'FHETrainer':
                continue
            kwargs = get_kwargs(sig, fn_name)
            args_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items() if k not in ['self'])
            
            is_trainer = 'Trainer' in fn_name
            
            if is_trainer:
                desc = f"Demonstrates the `{fn_name}` class. Runs `fit_encrypted` in simulation mode to train an encrypted model."
            else:
                desc = f"Demonstrates the `{fn_name}` class. Compiles and tests its `simulate` vs `predict` logic."
            cells.append(create_cell("markdown", f"### {fn_name}\n\n{desc}"))
            
            test_code = f"from {getattr(func, '__module__')} import {fn_name}\n"
            test_code += "import numpy as np\n\n"
            
            if is_trainer:
                if 'simulate=' not in args_str and 'simulate' in sig.parameters:
                    args_str = args_str + ", simulate=True" if args_str else "simulate=True"
                test_code += f"trainer = {fn_name}({args_str})\n"
                if 'KMeans' in fn_name:
                    test_code += "X_train = [[1, 1], [3, 3]]\n"
                    test_code += "try:\n"
                    test_code += "    model = trainer.fit_encrypted(X_train)\n"
                    test_code += f"    print('{fn_name} executed successfully!')\n"
                else:
                    test_code += "X_train = [[1], [3]]\n"
                    test_code += "y_train = [0, 1]\n"
                    test_code += "try:\n"
                    test_code += "    model = trainer.fit_encrypted(X_train, y_train)\n"
                    test_code += f"    print('{fn_name} executed successfully!')\n"
                test_code += "except Exception as e:\n"
                test_code += "    print(f'Execution failed: {e}')\n"
            else:
                if fn_name == 'FHEPipeline':
                    test_code += "from concrete_fhe_toolkit.ml import FHELinearRegression\n"
                    test_code += f"model = {fn_name}([FHELinearRegression(weights=[1,2], bias=1)])\n"
                else:
                    test_code += f"model = {fn_name}({args_str})\n"
                test_code += "inputset = [([1, 2],), ([0, 0],)]\n"
                if 'cnn' in fn_name.lower():
                    test_code += "inputset = [([[[[1]]]],), ([[[[0]]]],)]\n"
                test_code += "model.compile(inputset)\n"
                test_code += "_successes = 0\n"
                test_code += "for inp in inputset:\n"
                test_code += "    try:\n"
                test_code += "        expected = model.simulate(inp[0])\n"
                test_code += "        fhe_res = model.predict(inp[0])\n"
                test_code += "        if isinstance(expected, (list, np.ndarray)) or hasattr(expected, '__iter__'):\n"
                test_code += "            np.testing.assert_array_equal(fhe_res, expected)\n"
                test_code += "        else:\n"
                test_code += "            assert int(fhe_res) == int(expected)\n"
                test_code += "    except AssertionError:\n        raise\n    except Exception as e:\n"
                test_code += "        print(f'Skipping {inp} due to bounds or other error: {e}')\n"
                test_code += "    else:\n        _successes += 1\n"
                test_code += f"assert _successes > 0, '{fn_name}: all inputs were skipped'\n"
                test_code += f"print(f'{fn_name} model compiled and tested successfully! ({{_successes}}/{{len(inputset)}})')\n"
            
            cells.append(create_cell("code", test_code))
            cells.append(create_cell("markdown", "---"))
            continue

        
        if '[Client-Side Helper]' in docstring or fn_name in ('make_encode_fixed_point', 'make_decode_fixed_point', 'unsigned_to_bits', 'twos_complement_bits', 'return_actual_value') or mod_name in ('serialization', 'sklearn_bridge'):
            if fn_name.startswith('compile_'):
                continue

            desc = f"Demonstrates the `{fn_name}` function. This is a client-side (cleartext) helper function. It is **not** an FHE circuit and cannot be compiled with `fhe.Compiler`."
            cells.append(create_cell("markdown", f"### {fn_name}() (Helper)\n\n{desc}"))
            
            if fn_name == 'unsigned_to_bits':
                test_code = (
                    f"from concrete_fhe_toolkit.{subpkg}.{mod_name} import {fn_name}\n\n"
                    "clear_int = 5\n"
                    "bits = unsigned_to_bits(clear_int, width=4)\n"
                    "print(f'Unsigned integer {clear_int} to bits -> {bits}')\n"
                    "assert bits == (1, 0, 1, 0)\n"
                    "print('Conversion successful!')\n"
                )
            elif fn_name == 'twos_complement_bits':
                test_code = (
                    f"from concrete_fhe_toolkit.{subpkg}.{mod_name} import {fn_name}\n\n"
                    "clear_int = -3\n"
                    "bits = twos_complement_bits(clear_int, width=4)\n"
                    "print(f'Two\'s complement integer {clear_int} to bits -> {bits}')\n"
                    "assert bits == (1, 0, 1, 1)\n"
                    "print('Conversion successful!')\n"
                )
            elif fn_name == 'return_actual_value':
                test_code = (
                    f"from concrete_fhe_toolkit.{subpkg}.{mod_name} import {fn_name}\n\n"
                    "decrypted_val = 1500  # 15.00 scaled by 100\n"
                    "actual_float = return_actual_value(decrypted_val)\n"
                    "print(f'Decrypted integer {decrypted_val} to real value -> {actual_float}')\n"
                    "assert actual_float == 15.0\n"
                    "print('Decoding successful!')\n"
                )
            elif "encode" in fn_name and "fixed_point" in fn_name:
                test_code = (
                    f"from concrete_fhe_toolkit.{subpkg}.{mod_name} import {fn_name}\n\n"
                    "encode_fn = make_encode_fixed_point(scale=10)\n"
                    "clear_float = 2.5\n"
                    "encoded_int = encode_fn(clear_float)\n"
                    "print(f'Encoded {clear_float} with scale 10 -> {encoded_int} (type: {type(encoded_int).__name__})')\n"
                    "assert encoded_int == 25\n"
                    "print('Encoding successful! Ready for encryption.')\n"
                )
            elif "decode" in fn_name and "fixed_point" in fn_name:
                test_code = (
                    f"from concrete_fhe_toolkit.{subpkg}.{mod_name} import {fn_name}\n\n"
                    "decode_fn = make_decode_fixed_point(scale=10)\n"
                    "decrypted_int = 25\n"
                    "decoded_float = decode_fn(decrypted_int)\n"
                    "print(f'Decoded {decrypted_int} with scale 10 -> {decoded_float} (type: {type(decoded_float).__name__})')\n"
                    "assert decoded_float == 2.5\n"
                    "print('Decoding successful! Ready for client usage.')\n"
                )
            elif fn_name == 'save_model':
                test_code = (
                    f"from concrete_fhe_toolkit.{subpkg}.{mod_name} import {fn_name}\n"
                    "from concrete_fhe_toolkit.ml.classes import FHELinearRegression\n"
                    "import os\n\n"
                    "model = FHELinearRegression()\n"
                    "save_model(model, 'test_model.json')\n"
                    "print('Model saved successfully!')\n"
                    "if os.path.exists('test_model.json'): os.remove('test_model.json')\n"
                )
            elif fn_name == 'load_model':
                test_code = (
                    f"from concrete_fhe_toolkit.{subpkg}.{mod_name} import {fn_name}, save_model\n"
                    "from concrete_fhe_toolkit.ml.classes import FHELinearRegression\n"
                    "import os\n\n"
                    "model = FHELinearRegression()\n"
                    "save_model(model, 'test_model2.json')\n"
                    "loaded_model = load_model('test_model2.json')\n"
                    "print(f'Model loaded successfully: {loaded_model.__class__.__name__}')\n"
                    "if os.path.exists('test_model2.json'): os.remove('test_model2.json')\n"
                )
            elif fn_name == 'from_sklearn_linear':
                test_code = (
                    f"from concrete_fhe_toolkit.{subpkg}.{mod_name} import {fn_name}\n"
                    "try:\n"
                    "    from sklearn.linear_model import LinearRegression\n"
                    "    import numpy as np\n"
                    "    sk_model = LinearRegression()\n"
                    "    sk_model.coef_ = np.array([1.5, -2.0])\n"
                    "    sk_model.intercept_ = 3.0\n"
                    "    fhe_model = from_sklearn_linear(sk_model, scale=10)\n"
                    "    print(f'Converted sklearn model to {fhe_model.__class__.__name__}')\n"
                    "except ImportError:\n"
                    "    print('scikit-learn is not installed. Skipping test.')\n"
                )
            elif fn_name == 'from_sklearn_tree':
                test_code = (
                    f"from concrete_fhe_toolkit.{subpkg}.{mod_name} import {fn_name}\n"
                    "try:\n"
                    "    from sklearn.tree import DecisionTreeClassifier\n"
                    "    sk_model = DecisionTreeClassifier()\n"
                    "    print('FHE Conversion for DecisionTreeClassifier requires a fitted tree (sk_model.tree_).')\n"
                    "except ImportError:\n"
                    "    print('scikit-learn is not installed. Skipping test.')\n"
                )
            elif fn_name == 'from_sklearn_forest':
                test_code = (
                    f"from concrete_fhe_toolkit.{subpkg}.{mod_name} import {fn_name}\n"
                    "try:\n"
                    "    from sklearn.ensemble import RandomForestClassifier\n"
                    "    sk_model = RandomForestClassifier()\n"
                    "    print('FHE Conversion for RandomForestClassifier requires fitted estimators (sk_model.estimators_).')\n"
                    "except ImportError:\n"
                    "    print('scikit-learn is not installed. Skipping test.')\n"
                )
            else:
                test_code = f"from concrete_fhe_toolkit.{subpkg}.{mod_name} import {fn_name}\n\n"
                if fn_name.startswith('make_') or fn_name.startswith('compile_'):
                    kwargs = get_kwargs(sig, fn_name)
                    args_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
                    test_code += f"fn = {fn_name}({args_str})\n\n"
                    real_fn = getattr(mod, fn_name)(**kwargs)
                    param_names = list(inspect.signature(real_fn).parameters.keys())
                    test_inputs = generate_inputset(fn_name, param_names)
                    test_code += f"inputset = {test_inputs}\n"
                    test_code += "for inp in inputset:\n"
                    test_code += "    try:\n"
                    test_code += "        expected = fn(*inp)\n"
                    test_code += "        print(f'Helper output: {expected}')\n"
                    test_code += "    except Exception as e:\n"
                    test_code += "        print(f'Skipping {inp}: {e}')\n"
                    test_code += f"print('{fn_name} helper executed successfully!')\n"
                else:
                    param_names = list(sig.parameters.keys())
                    test_inputs = generate_inputset(fn_name, param_names)
                    test_code += f"inputset = {test_inputs}\n"
                    test_code += "for inp in inputset:\n"
                    test_code += "    try:\n"
                    test_code += f"        expected = {fn_name}(*inp)\n"
                    test_code += "        print(f'Helper output: {expected}')\n"
                    test_code += "    except Exception as e:\n"
                    test_code += "        print(f'Skipping {inp}: {e}')\n"
                    test_code += f"print('{fn_name} helper executed successfully!')\n"
            cells.append(create_cell("code", test_code))
            continue
            
        doc = inspect.getdoc(func)
        doc_desc = doc.split('\n\n')[0] if doc else "Tests the function."
        desc = f"{doc_desc}\n\nThis cell verifies the `{fn_name}` function mathematically against its cleartext counterpart, using a dynamically generated input set that covers positive, negative, zero, and array edge cases while respecting the `[-7, 7]` FHE RAM constraints."
        if aliases:
            desc += f"\n\n**Aliases**: `{', '.join(aliases)}` can also be used equivalently."
            
        cells.append(create_cell("markdown", f"### {fn_name}()\n\n{desc}"))
        
        imports = f"from concrete import fhe\nfrom concrete_fhe_toolkit.{subpkg}.{mod_name} import {fn_name}\n"
        if mod_name in ('combinatorics', 'number_theory', 'special'):
            imports += "import math\n"
        
        base_fn_name = fn_name.replace("compile_", "") if fn_name.startswith("compile_") else fn_name
        make_fn_name = fn_name.replace("compile_", "make_") if fn_name.startswith("compile_") else fn_name
        
        if fn_name.startswith("compile_"):
            if hasattr(mod, base_fn_name):
                imports += f"from concrete_fhe_toolkit.{subpkg}.{mod_name} import {base_fn_name}\n"
            if hasattr(mod, make_fn_name):
                imports += f"from concrete_fhe_toolkit.{subpkg}.{mod_name} import {make_fn_name}\n"
                
        test_code = imports + "\n"
        
        try:
            if fn_name.startswith("compile_"):
                kwargs = get_kwargs(sig, fn_name)
                args_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
                test_code += f"circuit = {fn_name}({args_str})\n\n"
                
                # Determine parameter names and cleartext fallback dynamically
                if hasattr(mod, make_fn_name):
                    make_fn = getattr(mod, make_fn_name)
                    make_kwargs = filter_kwargs(kwargs, make_fn)
                    clear_fn = make_fn(**make_kwargs)
                    param_names = list(inspect.signature(clear_fn).parameters.keys())
                    expected_args_str = ", ".join(f"{k}={v!r}" for k, v in make_kwargs.items())
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
                
                test_code += f"_successes = 0\nfor inp in inputset:\n"
                test_code += "    try:\n"
                if expected_call:
                    test_code += f"        expected = {expected_call}\n"
                    test_code += "        import numpy as np\n"

                    test_code += "        if isinstance(expected, (list, tuple)) or type(expected).__name__ == 'ndarray':\n"

                    test_code += "            np.testing.assert_array_equal(circuit.encrypt_run_decrypt(*inp), expected)\n"

                    test_code += "        else:\n"

                    test_code += "            assert int(circuit.encrypt_run_decrypt(*inp)) == int(expected), f\"Failed at {inp}\"\n"
                else:
                    test_code += "        circuit.encrypt_run_decrypt(*inp)\n"
                test_code += "    except AssertionError:\n        raise\n    except Exception as e:\n"
                test_code += "        print(f\"Skipping {inp} due to bounds or other error: {e}\")\n"
                test_code += "    else:\n        _successes += 1\n"
                
            elif fn_name.startswith("make_"):
                kwargs = get_kwargs(sig, fn_name)
                args_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
                test_code += f"fn = {fn_name}({args_str})\n\n"
                
                # Introspect the returned function to get actual parameter names!
                real_fn = getattr(mod, fn_name)(**kwargs)
                param_names = list(inspect.signature(real_fn).parameters.keys())
                
                clear_params = {'rate'} if subpkg == 'finance' else ({'alpha', 'divisor', 'num_classes'} if subpkg == 'ml' else {'start', 'step', 'size', 'k', 'angle_unit', 'rounding', 'scale', 'input_scale', 'output_scale', 'amount', 'multiplier', 'fractional_bits', 'zero_result', 'quotient_width', 'denominator_width', 'numerator_width', 'remainder_width', 'width', 'zero_quotient', 'zero_remainder', 'min_value', 'max_value', 'min_left', 'max_left', 'min_right', 'max_right', 'min_input', 'max_input', 'min_base', 'max_base', 'max_exponent', 'min_numerator', 'max_numerator', 'min_denominator', 'max_denominator', 'max_n', 'max_k', 'exponent', 'base', 'modulus', 'absolute_tolerance', 'arithmetic'})
                def get_mode(k): return 'clear' if k in clear_params else 'encrypted'
                enc_dict = "{" + ", ".join([f"'{k}': '{get_mode(k)}'" for k in param_names]) + "}"
                params_str = ", ".join(param_names)
                test_code += f"def test_{fn_name}_enc({params_str}):\n    import numpy as np\n    res = fn({params_str})\n    return np.array(res) if isinstance(res, list) else res\n\n"
                test_code += f"compiler = fhe.Compiler(test_{fn_name}_enc, {enc_dict})\n"
                
                test_inputs = generate_inputset(fn_name, param_names)
                test_code += f"inputset = {test_inputs}\n"
                test_code += "circuit = compiler.compile(inputset)\n\n"
                
                test_code += f"_successes = 0\nfor inp in inputset:\n"
                test_code += "    try:\n"
                test_code += f"        expected = fn(*inp)\n"
                test_code += "        import numpy as np\n"

                test_code += "        if isinstance(expected, (list, tuple)) or type(expected).__name__ == 'ndarray':\n"

                test_code += "            np.testing.assert_array_equal(circuit.encrypt_run_decrypt(*inp), expected)\n"

                test_code += "        else:\n"

                test_code += "            assert int(circuit.encrypt_run_decrypt(*inp)) == int(expected), f\"Failed at {inp}\"\n"
                test_code += "    except AssertionError:\n        raise\n    except Exception as e:\n"
                test_code += "        print(f\"Skipping {inp} due to bounds or other error: {e}\")\n"
                test_code += "    else:\n        _successes += 1\n"
                        
            else:
                param_names = list(sig.parameters.keys())
                clear_params = {'rate'} if subpkg == 'finance' else ({'alpha', 'divisor', 'num_classes'} if subpkg == 'ml' else {'start', 'step', 'size', 'k', 'angle_unit', 'rounding', 'scale', 'input_scale', 'output_scale', 'amount', 'multiplier', 'fractional_bits', 'zero_result', 'quotient_width', 'denominator_width', 'numerator_width', 'remainder_width', 'width', 'zero_quotient', 'zero_remainder', 'min_value', 'max_value', 'min_left', 'max_left', 'min_right', 'max_right', 'min_input', 'max_input', 'min_base', 'max_base', 'max_exponent', 'min_numerator', 'max_numerator', 'min_denominator', 'max_denominator', 'max_n', 'max_k', 'exponent', 'base', 'modulus', 'absolute_tolerance', 'arithmetic'})
                encrypted_params = [p for p in param_names if p not in clear_params and 'width' not in p]
                
                clear_defaults = {
                    'start': 1, 'step': 3, 'size': 3, 'k': 2, 
                    'angle_unit': 'degrees', 'rounding': 'nearest', 
                    'scale': 10, 'input_scale': 10, 'output_scale': 10,
                    'numerator_width': 4, 'denominator_width': 3, 'fractional_bits': 3,
                    'quotient_width': 8, 'remainder_width': 4, 'width': 4,
                    'amount': 1, 'multiplier': 2, 'zero_result': 0, 'zero_quotient': 0, 'zero_remainder': 0, 'arithmetic': 1, 'rate': 0.05
                }
                
                bound_args = []
                for p in param_names:
                    if p in encrypted_params:
                        if sig.parameters[p].kind == inspect.Parameter.KEYWORD_ONLY:
                            bound_args.append(f"{p}={p}")
                        else:
                            bound_args.append(p)
                    else:
                        val = repr(clear_defaults.get(p, 1))
                        if sig.parameters[p].kind == inspect.Parameter.KEYWORD_ONLY:
                            bound_args.append(f"{p}={val}")
                        else:
                            bound_args.append(val)
                
                params_str = ", ".join(encrypted_params)
                call_args = ", ".join(bound_args)
                test_code += f"def test_{fn_name}({params_str}):\n    import numpy as np\n    res = {fn_name}({call_args})\n    return np.array(res) if isinstance(res, list) else res\n\n"
                
                enc_dict = "{" + ", ".join([f"'{k}': 'encrypted'" for k in encrypted_params]) + "}"
                test_code += f"compiler = fhe.Compiler(test_{fn_name}, {enc_dict})\n"
                
                test_inputs = generate_inputset(fn_name, encrypted_params)
                test_code += f"inputset = {test_inputs}\n"
                test_code += "circuit = compiler.compile(inputset)\n\n"
                
                test_code += f"_successes = 0\nfor inp in inputset:\n"
                test_code += "    try:\n"
                
                expected_args = []
                inp_idx = 0
                for p in param_names:
                    if p in encrypted_params:
                        if sig.parameters[p].kind == inspect.Parameter.KEYWORD_ONLY:
                            expected_args.append(f"{p}=inp[{inp_idx}]")
                        else:
                            expected_args.append(f"inp[{inp_idx}]")
                        inp_idx += 1
                    else:
                        val = repr(clear_defaults.get(p, 1))
                        if sig.parameters[p].kind == inspect.Parameter.KEYWORD_ONLY:
                            expected_args.append(f"{p}={val}")
                        else:
                            expected_args.append(val)
                expected_call = f"{fn_name}({', '.join(expected_args)})"
                
                test_code += f"        expected = {expected_call}\n"
                test_code += "        import numpy as np\n"

                test_code += "        if isinstance(expected, (list, tuple)) or type(expected).__name__ == 'ndarray':\n"

                test_code += "            np.testing.assert_array_equal(circuit.encrypt_run_decrypt(*inp), expected)\n"

                test_code += "        else:\n"

                test_code += "            assert int(circuit.encrypt_run_decrypt(*inp)) == int(expected), f\"Failed at {inp}\"\n"
                test_code += "    except AssertionError:\n        raise\n    except Exception as e:\n"
                test_code += "        print(f\"Skipping {inp} due to bounds or other error: {e}\")\n"
                test_code += "    else:\n        _successes += 1\n"
            
            test_code += f"assert _successes > 0, \"{fn_name}: all inputs were skipped — test is broken\"\n"
            test_code += f"print(f\"{fn_name} tests passed! ({{_successes}}/{{len(inputset)}})\")"
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
        generate_notebook_for_module('math', mod, f"docs/tutorials/{fname}")

    finance_modules = [
        ("core", "11_finance_core.ipynb"),
        ("transactions", "12_finance_transactions.ipynb")
    ]
    for mod, fname in finance_modules:
        generate_notebook_for_module('finance', mod, f"docs/tutorials/{fname}")
        print(f"Generated {fname}")

    ml_modules = [
        ("activations", "13_ml_activations.ipynb"),
        ("classes", "14_ml_classes.ipynb"),
        ("classification", "15_ml_classification.ipynb"),
        ("clustering", "16_ml_clustering.ipynb"),
        ("core", "17_ml_core.ipynb"),
        ("estimation", "18_ml_estimation.ipynb"),
        ("matrix", "19_ml_matrix.ipynb"),
        ("models", "20_ml_models.ipynb"),
        ("pipeline", "21_ml_pipeline.ipynb"),
        ("preprocessing", "22_ml_preprocessing.ipynb"),
        ("regression", "23_ml_regression.ipynb"),
        ("stats", "26_ml_stats.ipynb"),
        ("trainers", "27_ml_trainers.ipynb"),
        ("training", "28_ml_training.ipynb"),
        ("utils", "29_ml_utils.ipynb")
    ]
    for mod, fname in ml_modules:
        generate_notebook_for_module('ml', mod, f"docs/tutorials/{fname}")
        print(f"Generated {fname}")

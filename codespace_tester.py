import json
import glob
import subprocess
import os
import sys

def test_notebooks():
    # Codespace paths for notebooks
    notebooks = sorted(glob.glob(r'docs/tutorials/*math*.ipynb'))
    if not notebooks:
        print("No notebooks found! Run this script from the root of the repository.")
        sys.exit(1)
        
    report = ["# Concrete FHE Notebooks - Automated Test Report\n"]
    
    os.makedirs('scratch', exist_ok=True)
    temp_script = 'scratch/temp_cell.py'
    
    for nb_path in notebooks:
        nb_name = os.path.basename(nb_path)
        report.append(f"## Testing {nb_name}\n")
        print(f"\nTesting {nb_name}...")
        
        with open(nb_path, encoding='utf-8') as f:
            nb = json.load(f)
            
        cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
        for i, cell in enumerate(cells):
            source = "".join(cell['source'])
            
            # Extract function name for reporting
            fn_name = "unknown"
            for line in source.split('\n'):
                if 'def test_' in line:
                    fn_name = line.split('def test_')[1].split('(')[0]
                    break
                elif line.startswith('circuit = compile_'):
                    fn_name = line.split('circuit = ')[1].split('(')[0]
                    break
                    
            print(f"  Running cell {i+1}/{len(cells)} (Function: {fn_name})...", end='', flush=True)
            
            with open(temp_script, 'w', encoding='utf-8') as f:
                f.write(source)
                
            try:
                # Setup python path to include src
                env = os.environ.copy()
                env['PYTHONPATH'] = 'src:' + env.get('PYTHONPATH', '')
                
                # Execute in an isolated subprocess per cell
                result = subprocess.run(
                    [sys.executable, temp_script],
                    capture_output=True,
                    text=True,
                    timeout=180, # 3 mins per cell limit to catch OOM/Infinite Loops
                    env=env
                )
                
                if result.returncode == 0:
                    print(" SUCCESS")
                    report.append(f"- ✅ `{fn_name}`: Passed")
                else:
                    print(" FAILED")
                    error_out = result.stderr.strip() if result.stderr.strip() else result.stdout.strip()
                    report.append(f"- ❌ **FAILED**: `{fn_name}` (Cell {i+1})\n")
                    report.append(f"  <details><summary>View Error Traceback</summary>\n\n  ```text\n  {error_out[-1500:]}\n  ```\n  </details>\n")
                    
            except subprocess.TimeoutExpired:
                print(" TIMEOUT")
                report.append(f"- ⚠️ **TIMEOUT**: `{fn_name}` (Cell {i+1}) - Exceeded 3 minutes (RAM/OOM protection).")
            except Exception as e:
                print(" ERROR")
                report.append(f"- ❌ **ERROR**: `{fn_name}` - {str(e)}")
                
        report.append("\n---\n")
        
    report_path = 'test_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report))
        
    print(f"\nTesting complete! Full report saved to {report_path}")

if __name__ == "__main__":
    test_notebooks()

from concrete import fhe

def test_eq(x, y):
    return x == y

compiler = fhe.Compiler(test_eq, {"x": "encrypted", "y": "encrypted"})
inputset = [(1, 1), (1, 2), (0, 0), (-1, -1), (-5, 5)]
circuit = compiler.compile(inputset)

print("1 == 1:", circuit.encrypt_run_decrypt(1, 1))
print("1 == 2:", circuit.encrypt_run_decrypt(1, 2))
print("0 == 0:", circuit.encrypt_run_decrypt(0, 0))
print("-1 == -1:", circuit.encrypt_run_decrypt(-1, -1))

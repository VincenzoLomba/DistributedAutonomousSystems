import sympy as sp

# Dimensione della matrice
n = 2  

# Variabile simbolica per s
s = sp.symbols('s')

# Creazione delle matrici simboliche per i blocchi
A11 = sp.MatrixSymbol('A11', n, n)
A12 = sp.MatrixSymbol('A12', n, n)
A13 = sp.MatrixSymbol('A13', n, n)
A31 = sp.MatrixSymbol('A31', n, n)
A32 = sp.MatrixSymbol('A32', n, n)

# Definire la matrice A come matrice a blocchi
zero_matrix = sp.zeros(n, n)
A = sp.BlockMatrix([
    [A11, A12, A13],
    [zero_matrix, zero_matrix, zero_matrix],
    [A31, A32, zero_matrix]
])

# Creazione della matrice sI - A
I = sp.eye(A.shape[0])
sI_A = s * I - A

# Calcolare l'inversa di sI - A
try:
    sI_A_inv = sI_A.inv()
    # Semplificare l'inversa
    sI_A_inv_simplified = sp.simplify(sI_A_inv)
    sp.pprint(sI_A_inv_simplified, use_unicode=True)
except Exception as e:
    print(f"Errore nel calcolo dell'inversa: {e}")

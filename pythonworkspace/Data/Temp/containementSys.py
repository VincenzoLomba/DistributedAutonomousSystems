import sympy as sp

# Dimensione dei blocchi (es. 2x2 per semplicità, puoi aumentare)
n = 2

# Crea simboli per il tempo
t, tau = sp.symbols('t tau', real=True)

# Blocchi simbolici
A11 = sp.MatrixSymbol('A11', n, n)
A12 = sp.MatrixSymbol('A12', n, n)
A13 = sp.MatrixSymbol('A13', n, n)
A31 = sp.MatrixSymbol('A31', n, n)
A32 = sp.MatrixSymbol('A32', n, n)

# Vettori iniziali
x1_0 = sp.MatrixSymbol('x1_0', n, 1)
x2_0 = sp.MatrixSymbol('x2_0', n, 1)
x3_0 = sp.MatrixSymbol('x3_0', n, 1)

# Matrice blocco ridotta Ar
Ar = sp.Matrix([[A11, A13],
                [A31, sp.ZeroMatrix(n, n)]])

# Termine forzante B * x2(0)
B = sp.Matrix([[A12],
               [A32]])

# Stato iniziale ridotto
x_r0 = sp.Matrix.vstack(x1_0, x3_0)

# Soluzione omogenea: e^{Ar t} x_r(0)
x_r_hom = (Ar * t).exp() * x_r0

# Integrale della risposta al forzamento costante
integrand = (Ar * (t - tau)).exp() * B
integral = sp.integrate(integrand, (tau, 0, t))

# Risposta completa
x_r = x_r_hom + integral * x2_0

# Ricostruzione dello stato completo: x(t) = [x1(t); x2(0); x3(t)]
x_t = sp.Matrix.vstack(x_r[:n, :], x2_0, x_r[n:, :])
x_t


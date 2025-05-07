
import numpy as np

def gradientTrackingMethod(A, stepsize, localCostFunctions, decisionVariableInitialValue, maxIters, tolerance):

    N = A.shape[0]
    decisionVariableDimension = decisionVariableInitialValue.shape[1] # prendere la maggiore
    # d = decisionVariableDimension/N
    z = decisionVariableInitialValue # np.zeros((maxIters, N, decisionVariableDimension))
    s = np.zeros((maxIters, N, decisionVariableDimension))
    for i in range(N):
        s[0, i, :] = localCostFunctions[i](decisionVariableInitialValue)[1]
    cst = np.zeros((maxIters, N))
    grd = np.zeros((maxIters, N, decisionVariableDimension))

    for k in range(maxIters - 1):

        for i in range(N):
            # z[k + 1, i, :] = A[i, i] * z[k, i, :]
            N_i = np.nonzero(A[i])[0] # localizzare
            for j in N_i:
                z[k + 1, i, :] += A[i, j] * z[k, j, :] # prodotto per scalare

            z[k + 1, i, :] -= stepsize * s[k, i, :]

        for i in range(N):
            s[k + 1, i, :] = A[i, i] * s[k, i, :]
            N_i = np.nonzero(A[i])[0] # localizzare
            for j in N_i:
                s[k + 1, i, :] += A[i, j] * s[k, j, :] # prodotto per scalare

            cstFun = localCostFunctions[i]
            cstzk, grdzk = cstFun(z[k, i])
            s[k + 1, i, :] += (cstFun(z[k + 1, i])[1] - grdzk)
            cst[k, i] = cstzk
            grd[k, i, :] = grdzk

        # chiedere a notar
        # se fermarsi con la norma de[i] gradient[i] o con la norma di z[k + 1] - z[k]
        finish = True
        for i in range(N):
            if np.linalg.norm(grd[k, i, :]) > tolerance:
                finish = False
                break
        if finish:
            print("Converged at iteration", k)
            break

        res = GTMSolution(maxIters, N, decisionVariableDimension)
        res.z = z
        res.s = s
        res.cst = cst
        res.grd = grd
        return res

class GTMSolution:

    def __init__(self, maxIters, N, decisionVariableDimension):
        self.N = N
        self.maxIters = maxIters
        self.decisionVariableDimension = decisionVariableDimension
        z = np.zeros((maxIters, N, decisionVariableDimension))
        s = np.zeros((maxIters, N, decisionVariableDimension))
        cst = np.zeros((maxIters, N))
        grd = np.zeros((maxIters, N, decisionVariableDimension))


            
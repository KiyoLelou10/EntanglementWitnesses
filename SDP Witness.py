import numpy as np
import secrets
import itertools
from typing import List, Tuple, Dict
import cvxpy as cp

# =========================
# Pauli plumbing
# =========================

I2 = np.array([[1, 0], [0, 1]], dtype=complex)
X2 = np.array([[0, 1], [1, 0]], dtype=complex)
Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z2 = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = {"I": I2, "X": X2, "Y": Y2, "Z": Z2}

def pauli_string_to_matrix(pstr: str) -> np.ndarray:
    M = np.array([[1]], dtype=complex)
    for ch in pstr:
        M = np.kron(M, PAULI[ch])
    return M

def all_pauli_strings(n: int) -> List[str]:
    return ["".join(t) for t in itertools.product("IXYZ", repeat=n)]

# =========================
# Quantum helpers
# =========================

def random_pure_state(dim: int, rng: np.random.Generator) -> np.ndarray:
    psi = rng.normal(size=(dim,)) + 1j * rng.normal(size=(dim,))
    psi /= np.linalg.norm(psi)
    return psi

def density_matrix_from_state(psi: np.ndarray) -> np.ndarray:
    return np.outer(psi, psi.conj())

def partial_transpose_matrix(M: np.ndarray, A_qubits: List[int], n_qubits: int) -> np.ndarray:
    """
    Partial transpose on subsystem A (qubit indices in A_qubits) for an n-qubit operator M.
    Implemented by reshaping into (row bits..., col bits...) and swapping row/col indices for qubits in A.
    """
    d = 2 ** n_qubits
    assert M.shape == (d, d)
    A = set(A_qubits)

    T = M.reshape([2] * n_qubits + [2] * n_qubits)
    axes = list(range(2 * n_qubits))
    for i in A:
        axes[i], axes[i + n_qubits] = axes[i + n_qubits], axes[i]
    Tpt = np.transpose(T, axes=axes)
    return Tpt.reshape(d, d)

# =========================
# SDP: optimal decomposable witness (P=0)
# =========================

def solve_optimal_decomposable_witness(
    rho: np.ndarray,
    A_qubits: List[int],
    solver_iters: int = 5000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve:
        minimize    Tr(Q rho^{T_A})
        subject to  Q >= 0, Tr(Q)=1
    and output W = Q^{T_A}.
    """
    n_qubits = int(np.log2(rho.shape[0]))
    d = 2 ** n_qubits

    rho_TA = partial_transpose_matrix(rho, A_qubits=A_qubits, n_qubits=n_qubits)
    rho_TA = 0.5 * (rho_TA + rho_TA.conj().T)  # numeric hermitization

    Q = cp.Variable((d, d), hermitian=True)

    constraints = [
        Q >> 0,
        cp.real(cp.trace(Q)) == 1.0,
    ]

    objective = cp.Minimize(cp.real(cp.trace(Q @ rho_TA)))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, max_iters=solver_iters, eps=1e-6, verbose=True)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Solver status: {prob.status}")

    Q_opt = Q.value
    Q_opt = 0.5 * (Q_opt + Q_opt.conj().T)

    W_opt = partial_transpose_matrix(Q_opt, A_qubits=A_qubits, n_qubits=n_qubits)
    W_opt = 0.5 * (W_opt + W_opt.conj().T)

    return W_opt, Q_opt

# =========================
# Full Pauli decomposition
# =========================

def pauli_decompose_full(W: np.ndarray, n_qubits: int) -> Dict[str, float]:
    """
    Decompose W into the full Pauli basis:
        W = sum_s c_s P_s
    with c_s = Tr(P_s W)/d (since Tr(P_s P_t) = d delta_{s,t}).
    Returns real coefficients (imag parts are numerical noise).
    """
    d = 2 ** n_qubits
    coeffs: Dict[str, float] = {}

    for s in all_pauli_strings(n_qubits):
        P = pauli_string_to_matrix(s)
        c = np.trace(P @ W) / d
        coeffs[s] = float(np.real(c))  # should be real for Hermitian W

    return coeffs

# =========================
# Demo main
# =========================

def main():
    n_qubits = 6
    d = 2 ** n_qubits

    # Fixed 3|3 cut: first three qubits in A
    A_qubits = [0, 1, 2]
    B_qubits = [3, 4, 5]
    print(f"A_qubits={A_qubits} | B_qubits={B_qubits}")

    # Random state
    seed_state = secrets.randbits(64)
    print(f"seed_state={seed_state}")
    rng = np.random.default_rng(seed_state)

    psi = random_pure_state(d, rng)
    rho = density_matrix_from_state(psi)

    # Solve SDP
    W_opt, Q_opt = solve_optimal_decomposable_witness(rho, A_qubits=A_qubits, solver_iters=5000)

    # Evaluate witness
    tr_W_rho = float(np.real(np.trace(W_opt @ rho)))
    evals_Q = np.linalg.eigvalsh(Q_opt)
    print("\n=== Learned witness (full operator space, decomposable, P=0) ===")
    print(f"Tr(W rho)  = {tr_W_rho:.6e}")
    print(f"min eig(Q) = {float(np.min(evals_Q)):.6e}")
    print(f"Tr(Q)      = {float(np.real(np.trace(Q_opt))):.6f}")

    # Reference PPT test (cheating)
    rho_TA = partial_transpose_matrix(rho, A_qubits=A_qubits, n_qubits=n_qubits)
    rho_TA = 0.5 * (rho_TA + rho_TA.conj().T)
    lam_min = float(np.min(np.linalg.eigvalsh(rho_TA)))
    print("\n=== Reference (PPT test) ===")
    print(f"min eigenvalue of rho^(T_A) = {lam_min:.6e}")
    print(f"Conclusion from W: {'ENTANGLED' if tr_W_rho < 0 else 'inconclusive'}")
    print(f"Conclusion from PPT: {'NPT entangled' if lam_min < 0 else 'PPT (not detected by PPT)'}")

    # Full Pauli decomposition of W
    print("\n=== Full Pauli decomposition of W (top 20 by |coeff|) ===")
    coeffs = pauli_decompose_full(W_opt, n_qubits=n_qubits)
    top = sorted(coeffs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:20]
    for s, c in top:
        print(f"{s}: {c:+.6e}")

    # How dense is it?
    thresh = 1e-6
    nnz = sum(1 for v in coeffs.values() if abs(v) > thresh)
    print(f"\n#coeffs with |c| > {thresh:g}: {nnz} / {len(coeffs)}")

if __name__ == "__main__":
    main()

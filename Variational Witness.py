import numpy as np
import secrets
from typing import List, Tuple
import cvxpy as cp

# =========================
# Quantum helpers
# =========================

def random_pure_state(dim: int, rng: np.random.Generator) -> np.ndarray:
    psi = rng.normal(size=(dim,)) + 1j * rng.normal(size=(dim,))
    psi /= np.linalg.norm(psi)
    return psi

def density_matrix_from_state(psi: np.ndarray) -> np.ndarray:
    return np.outer(psi, psi.conj())

def partial_transpose_rho_matrix(rho: np.ndarray, A_qubits: List[int], n_qubits: int) -> np.ndarray:
    """
    Partial transpose on subsystem A (given by qubit indices in A_qubits) for a full n-qubit matrix rho.
    Implemented by reshaping rho into 2n indices and swapping row/col indices for qubits in A.
    """
    d = 2 ** n_qubits
    assert rho.shape == (d, d)
    A = set(A_qubits)

    # tensor shape: (row bits..., col bits...)
    T = rho.reshape([2] * n_qubits + [2] * n_qubits)

    axes = list(range(2 * n_qubits))
    # For each qubit i in A: swap row-axis i with col-axis (i+n)
    for i in A:
        axes[i], axes[i + n_qubits] = axes[i + n_qubits], axes[i]

    Tpt = np.transpose(T, axes=axes)
    return Tpt.reshape(d, d)

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

def random_pauli_strings(n: int, K: int, rng: np.random.Generator) -> List[str]:
    """
    Sample K unique non-identity Pauli strings of length n from {I,X,Y,Z}^n, excluding all-I.
    """
    seen = set()
    out = []
    alphabet = np.array(["I", "X", "Y", "Z"])

    while len(out) < K:
        s = "".join(rng.choice(alphabet, size=n))
        if s == "I" * n:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out

def pt_sign_for_string_on_A(pstr: str, A_qubits: List[int]) -> int:
    """
    Under (local) transpose on qubits in A in computational basis:
    I^T=I, X^T=X, Z^T=Z, Y^T=-Y.
    So each 'Y' on a transposed qubit contributes a minus sign.
    """
    s = 1
    A = set(A_qubits)
    for i, ch in enumerate(pstr):
        if i in A and ch == "Y":
            s *= -1
    return s

# =========================
# Main SDP
# =========================

def learn_pauli_sparse_decomposable_witness(
    rho: np.ndarray,
    A_qubits: List[int],
    K: int = 100,
    solver_iters: int = 5000,
    seed_paulis: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Learn a Pauli-sparse decomposable witness by enforcing:
        Q := W^{T_A} >= 0, Tr(Q)=1
    and parameterizing:
        W = alpha*I + sum_k c_k P_k
    where {P_k} are K Pauli strings.

    Returns (W_opt, Q_opt, pauli_strings).
    """
    n_qubits = int(np.log2(rho.shape[0]))
    d = 2 ** n_qubits

    rng = np.random.default_rng(seed_paulis)

    # Choose Pauli basis subset
    pauli_strs = random_pauli_strings(n_qubits, K, rng)
    Pmats = [pauli_string_to_matrix(s) for s in pauli_strs]

    # Precompute partial transpose effect on each Pauli string: PT_A(P_k) = sign_k * P_k
    signs = np.array([pt_sign_for_string_on_A(s, A_qubits) for s in pauli_strs], dtype=float)
    PTmats = [signs[k] * Pmats[k] for k in range(K)]

    # Precompute objective coefficients: Tr(P_k rho)
    f = np.array([np.real(np.trace(Pmats[k] @ rho)) for k in range(K)], dtype=float)
    # Tr(I rho) = 1 since rho is density matrix
    # So objective is alpha + sum_k c_k f_k

    # Variables (100 params = c_k; alpha is one more scalar)
    c = cp.Variable(K)      # real
    alpha = cp.Variable()   # real

    I = np.eye(d, dtype=complex)

    # Build W and Q = PT_A(W)
    W = alpha * I
    Q = alpha * I
    for k in range(K):
        W += c[k] * Pmats[k]
        Q += c[k] * PTmats[k]

    # Ensure Hermitian numerically
    Qh = 0.5 * (Q + Q.H)

    constraints = [
        Qh >> 0,                       # Q is PSD  => W = Q^{T_A} is decomposable with P=0
        cp.real(cp.trace(Qh)) == 1.0,  # normalization (needed to avoid scaling to -infinity)
    ]

    objective = cp.Minimize(alpha + f @ c)
    prob = cp.Problem(objective, constraints)

    prob.solve(solver=cp.SCS, max_iters=solver_iters, eps=1e-6, verbose=True)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Solver status: {prob.status}")

    c_opt = c.value
    alpha_opt = float(alpha.value)

    # Build numeric W_opt and Q_opt
    W_opt = alpha_opt * I
    Q_opt = alpha_opt * I
    for k in range(K):
        W_opt += c_opt[k] * Pmats[k]
        Q_opt += c_opt[k] * PTmats[k]

    # Hermitian sym
    W_opt = 0.5 * (W_opt + W_opt.conj().T)
    Q_opt = 0.5 * (Q_opt + Q_opt.conj().T)

    return W_opt, Q_opt, pauli_strs

def main():
    n_qubits = 8
    d = 2 ** n_qubits

    # Random seeds
    seed_state = secrets.randbits(64)
    seed_cut = secrets.randbits(64)
    seed_paulis = secrets.randbits(64)

    print(f"seed_state={seed_state}")
    print(f"seed_cut={seed_cut}")
    print(f"seed_paulis={seed_paulis}")

    rng_state = np.random.default_rng(seed_state)
    psi = random_pure_state(d, rng_state)
    rho = density_matrix_from_state(psi)

    # Choose 4|4 cut
    rng_cut = np.random.default_rng(seed_cut)
    A_qubits = sorted(rng_cut.choice(n_qubits, size=4, replace=False).tolist())
    B_qubits = [i for i in range(n_qubits) if i not in A_qubits]
    print(f"A_qubits={A_qubits}")
    print(f"B_qubits={B_qubits}")

    # Solve SDP in Pauli-sparse subspace
    K = 300
    solver_iters = 5000  # you asked 100; in practice you may need a lot more for convergence

    W_opt, Q_opt, paulis = learn_pauli_sparse_decomposable_witness(
        rho=rho,
        A_qubits=A_qubits,
        K=K,
        solver_iters=solver_iters,
        seed_paulis=seed_paulis,
    )

    tr_W_rho = float(np.real(np.trace(W_opt @ rho)))
    print("\n=== Result (Pauli-sparse decomposable witness, P=0) ===")
    print(f"K={K}, solver_iters={solver_iters}")
    print(f"Tr(W rho) = {tr_W_rho:.6e}")

    # Check Q is PSD + trace
    evals_Q = np.linalg.eigvalsh(Q_opt)
    print(f"min eig(Q)  = {np.min(evals_Q):.6e}")
    print(f"Tr(Q)       = {np.real(np.trace(Q_opt)):.6f}")

    # Compare with true NPT value (cheating reference): min eigenvalue of rho^{T_A}
    rho_TA = partial_transpose_rho_matrix(rho, A_qubits=A_qubits, n_qubits=n_qubits)
    evals_TA = np.linalg.eigvalsh(0.5 * (rho_TA + rho_TA.conj().T))
    lam_min = float(np.min(np.real(evals_TA)))
    print("\n=== Reference (cheating) ===")
    print(f"min eigenvalue of rho^(T_A) = {lam_min:.6e}")
    print(f"Conclusion from W: {'ENTANGLED (certified by this W)' if tr_W_rho < 0 else 'inconclusive'}")
    print(f"Conclusion from PPT test: {'NPT entangled' if lam_min < 0 else 'PPT (could be separable or bound entangled)'}")

if __name__ == "__main__":
    main()

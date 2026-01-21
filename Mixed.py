import numpy as np
import secrets
from dataclasses import dataclass
from typing import List, Optional, Tuple

# =========================
# Helpers: quantum plumbing
# =========================

def random_pure_state(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-ish random complex state via i.i.d. complex Gaussian, normalized."""
    psi = rng.normal(size=(dim,)) + 1j * rng.normal(size=(dim,))
    psi /= np.linalg.norm(psi)
    return psi

def density_matrix_from_state(psi: np.ndarray) -> np.ndarray:
    """rho = |psi><psi|"""
    return np.outer(psi, psi.conj())

def inverse_permutation(perm: List[int]) -> List[int]:
    inv = [0] * len(perm)
    for new_pos, old_pos in enumerate(perm):
        inv[old_pos] = new_pos
    return inv

def permute_density_matrix_qubits(rho: np.ndarray, perm: List[int], n_qubits: int) -> np.ndarray:
    """
    Reorder qubits of an n-qubit operator (density matrix) according to perm.

    perm is a list of length n_qubits describing the NEW ordering in terms of OLD indices.
    Example: perm=[2,0,1] means new qubit-0 is old qubit-2, new qubit-1 is old qubit-0, etc.
    """
    d = 2 ** n_qubits
    if rho.shape != (d, d):
        raise ValueError(f"rho must be shape {(d, d)} for n_qubits={n_qubits}, got {rho.shape}")
    if sorted(perm) != list(range(n_qubits)):
        raise ValueError("perm must be a permutation of [0..n_qubits-1]")

    # Tensor view: (row qubits..., col qubits...)
    rho_t = rho.reshape([2] * n_qubits + [2] * n_qubits)

    # Permute row-qubit axes and column-qubit axes in the same way
    row_axes = perm
    col_axes = [p + n_qubits for p in perm]
    rho_tp = np.transpose(rho_t, axes=row_axes + col_axes)

    return rho_tp.reshape(d, d)

def partial_transpose_bipartite(rho: np.ndarray, dA: int, dB: int, sys: str = "A") -> np.ndarray:
    """
    Partial transpose on subsystem A or B for a bipartite operator on C^{dA} ⊗ C^{dB}.

    Assumes the matrix is already in A|B ordering (i.e., basis index = i*dB + alpha).
    """
    d = dA * dB
    if rho.shape != (d, d):
        raise ValueError(f"rho must be shape {(d, d)} for dims {(dA, dB)}, got {rho.shape}")

    X = rho.reshape(dA, dB, dA, dB)  # indices (i,alpha, j,beta)

    sys = sys.upper()
    if sys == "A":
        # (i,alpha, j,beta) -> (j,alpha, i,beta)
        X_pt = X.transpose(2, 1, 0, 3)
    elif sys == "B":
        # (i,alpha, j,beta) -> (i,beta, j,alpha)
        X_pt = X.transpose(0, 3, 2, 1)
    else:
        raise ValueError("sys must be 'A' or 'B'")

    return X_pt.reshape(d, d)

def is_psd_hermitian(X: np.ndarray, tol: float = 1e-10) -> bool:
    Xh = 0.5 * (X + X.conj().T)
    evals = np.linalg.eigvalsh(Xh)
    return np.min(evals) >= -tol


# =========================
# Result container
# =========================

@dataclass
class WitnessResult:
    n_qubits: int
    a: int
    b: int
    A_qubits: List[int]
    B_qubits: List[int]
    perm_to_AB: List[int]
    min_eig_rho_TA: float
    expectation_TrWrho: float
    entangled_npt: bool
    P: np.ndarray
    Q: np.ndarray
    W_in_AB_order: np.ndarray
    W_in_original_order: np.ndarray


# =========================
# Core: best decomposable witness for fixed cut
# =========================

def best_decomposable_witness_for_rho(
    rho: np.ndarray,
    a: int,
    b: int,
    A_qubits: Optional[List[int]] = None,
    seed_for_partition: Optional[int] = None,
    tol: float = 1e-10,
) -> WitnessResult:
    """
    For a fixed rho on n=a+b qubits, find an optimal decomposable witness of the form
      W = P + Q^{T_A},  P>=0, Q>=0
    under the standard normalization Tr(Q)=1, Tr(P)=0 at optimum.

    Returns P, Q, W, and Tr(W rho). If Tr(W rho) < 0 then rho is NPT-entangled across that cut.
    """
    n = a + b
    d = 2 ** n
    if rho.shape != (d, d):
        raise ValueError(f"rho must be {(d,d)} for n={n}, got {rho.shape}")
    if n != 8:
        raise ValueError(f"This script expects a+b=8; got a+b={n}")

    rng = np.random.default_rng(seed_for_partition)

    # Choose A qubits
    if A_qubits is None:
        A_qubits = sorted(rng.choice(n, size=a, replace=False).tolist())
    else:
        A_qubits = list(A_qubits)
        if len(A_qubits) != a:
            raise ValueError(f"A_qubits must have length a={a}")
        if len(set(A_qubits)) != a:
            raise ValueError("A_qubits must not contain duplicates")
        if any(q < 0 or q >= n for q in A_qubits):
            raise ValueError("A_qubits indices out of range")

    B_qubits = [q for q in range(n) if q not in A_qubits]

    # Permute rho so that subsystem order is A|B (A qubits first)
    perm_to_AB = A_qubits + B_qubits
    rho_AB = permute_density_matrix_qubits(rho, perm_to_AB, n_qubits=n)

    dA, dB = 2 ** a, 2 ** b

    # Compute partial transpose of rho on A
    rho_TA = partial_transpose_bipartite(rho_AB, dA=dA, dB=dB, sys="A")

    # Solve analytically: min eigenvalue of rho^{T_A}
    evals, evecs = np.linalg.eigh(0.5 * (rho_TA + rho_TA.conj().T))
    idx = int(np.argmin(evals))
    lam_min = float(np.real(evals[idx]))
    v_min = evecs[:, idx]  # normalized

    # Construct Q = |v_min><v_min|  (PSD, Tr=1)
    Q = np.outer(v_min, v_min.conj())

    # Optimal P=0 for this objective
    P = np.zeros_like(Q)

    # Witness in A|B order: W = P + Q^{T_A} = Q^{T_A}
    W_AB = partial_transpose_bipartite(Q, dA=dA, dB=dB, sys="A")

    # Evaluate expectation Tr(W rho) in the same A|B basis
    tr_W_rho = float(np.real(np.trace(W_AB @ rho_AB)))

    # Transform witness back to original qubit order (optional)
    inv = inverse_permutation(perm_to_AB)
    W_orig = permute_density_matrix_qubits(W_AB, inv, n_qubits=n)

    entangled_npt = (lam_min < -tol)

    if not is_psd_hermitian(Q, tol=1e-8):
        raise RuntimeError("Q is unexpectedly not PSD (numerical issue).")

    return WitnessResult(
        n_qubits=n,
        a=a,
        b=b,
        A_qubits=A_qubits,
        B_qubits=B_qubits,
        perm_to_AB=perm_to_AB,
        min_eig_rho_TA=lam_min,
        expectation_TrWrho=tr_W_rho,
        entangled_npt=entangled_npt,
        P=P,
        Q=Q,
        W_in_AB_order=W_AB,
        W_in_original_order=W_orig,
    )


# =========================
# Demo run (8 qubits) with noisy mixed state
# =========================

def main():
    n_qubits = 8

    # User-settable partition sizes
    a = 4
    b = 4
    if a + b != n_qubits:
        raise ValueError(f"Need a+b=8, got {a+b}")

    # Noise mixing parameter:
    # rho = p * |psi><psi| + (1-p) * I/dim
    # Try p ~ 0.05..0.5 to see both regimes.
    p = 0.5
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0,1]")

    # Cryptographically strong OS entropy for seeds (new each run)
    seed_state = secrets.randbits(64)
    seed_partition = secrets.randbits(64)

    # Print seeds so you can reproduce a run later if you want
    print(f"seed_state={seed_state}")
    print(f"seed_partition={seed_partition}")
    print(f"mixing p={p}")

    rng = np.random.default_rng(seed_state)

    dim = 2 ** n_qubits
    psi = random_pure_state(dim, rng)
    rho_pure = density_matrix_from_state(psi)

    # Noisy mixed state: depolarized pure state
    I = np.eye(dim, dtype=complex)
    rho = p * rho_pure + (1.0 - p) * (I / dim)

    # (Optional) numerical hygiene
    rho = 0.5 * (rho + rho.conj().T)
    rho /= np.trace(rho).real  # ensure trace=1

    res = best_decomposable_witness_for_rho(
        rho=rho,
        a=a,
        b=b,
        A_qubits=None,
        seed_for_partition=seed_partition,
        tol=1e-10,
    )

    print("=== Random 8-qubit NOISY mixed state entanglement test with decomposable witness ===")
    print(f"Partition sizes: |A|={res.a}, |B|={res.b}")
    print(f"A qubits: {res.A_qubits}")
    print(f"B qubits: {res.B_qubits}")
    print()
    print(f"min eigenvalue of rho^(T_A): {res.min_eig_rho_TA:.6e}")
    print(f"Tr(W rho):                  {res.expectation_TrWrho:.6e}")
    print(f"Conclusion (for this cut):  {'ENTANGLED (NPT)' if res.entangled_npt else 'inconclusive (PPT or separable)'}")
    print()
    print("Witness form: W = P + Q^(T_A) with P>=0, Q>=0")
    print(f"Tr(Q) = {np.trace(res.Q).real:.6f}  (should be 1.0)")
    print(f"P is zero matrix: {np.allclose(res.P, 0)}")

if __name__ == "__main__":
    main()

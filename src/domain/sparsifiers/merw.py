from __future__ import annotations

import numpy as np
import scipy.sparse.linalg as sla


def _dominant_eigenvector(A: np.ndarray | any) -> np.ndarray:
    """
    returns the dominant eigenvector of an adjacency matrix.
    """
    try:
        eigenvalues, eigenvectors = sla.eigsh(A, k=1, which="LM")
        v = np.abs(eigenvectors[:, 0])
    except Exception:
        dense = A.toarray() if hasattr(A, "toarray") else A
        eigenvalues, eigenvectors = np.linalg.eigh(dense)
        v = np.abs(eigenvectors[:, -1])
    return v / np.linalg.norm(v)

def _stationary_distribution(psi: np.ndarray) -> np.ndarray:
    """
    returns the stationary distribution proportional to psi^2.
    """
    p = psi ** 2
    total = p.sum()
    return p / total if total > 0 else p
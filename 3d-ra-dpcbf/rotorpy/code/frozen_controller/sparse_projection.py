"""Reusable sparse projection QP shared by hard-constraint controllers.

Solves min 0.5 ||u-target||^2 subject to B u + C >= 0 and -2 <= u <= 2.
The mathematical problem is identical to dynamic_study.kernel.project_qp;
only the numerical backend changes to prevent worst-case active-set enumeration.
"""
import numpy as np


class SparseProjectionQP:
    def __init__(self, n):
        import osqp
        import scipy.sparse as sp
        self.n = int(n); self.sp = sp; self.fallbacks = 0
        rows = []; cols = []; values = []
        for j in range(3):
            rows.extend(range(self.n)); cols.extend([j] * self.n); values.extend([1.0] * self.n)
            rows.append(self.n + j); cols.append(j); values.append(1.0)
        self.rows = np.asarray(rows, dtype=np.int64); self.cols = np.asarray(cols, dtype=np.int64)
        pattern = sp.csc_matrix((np.asarray(values), (self.rows, self.cols)), shape=(self.n + 3, 3))
        self.nnz = int(pattern.nnz)
        if self.nnz != 3 * self.n + 3: raise RuntimeError("Unexpected hard-QP sparsity")
        self.solver = osqp.OSQP()
        self.solver.setup(P=sp.eye(3, format="csc"), q=np.zeros(3), A=pattern,
                          l=np.full(self.n + 3, -np.inf), u=np.full(self.n + 3, np.inf),
                          verbose=False, eps_abs=1e-9, eps_rel=1e-9, max_iter=20000,
                          polishing=False, adaptive_rho=True)

    def _matrix(self, B):
        values = []
        for j in range(3): values.extend(B[:, j]); values.append(1.0)
        matrix = self.sp.csc_matrix((np.asarray(values), (self.rows, self.cols)), shape=(self.n + 3, 3))
        if matrix.nnz != self.nnz: raise RuntimeError("Hard-QP sparsity changed")
        return matrix

    def solve(self, target, B, C, force_fallback=False):
        target = np.asarray(target, float); B = np.asarray(B, float); C = np.asarray(C, float)
        if B.shape != (self.n, 3) or C.shape != (self.n,): raise ValueError("Hard-QP shape mismatch")
        matrix = self._matrix(B)
        lower = np.r_[-C, [-2.0] * 3]; upper = np.r_[[np.inf] * self.n, [2.0] * 3]
        self.solver.update(q=-target, Ax=matrix.data, l=lower, u=upper)
        result = self.solver.solve(raise_error=False)
        good = (not force_fallback) and result.info.status_val in (1, 2) and result.x is not None
        if good:
            u = np.asarray(result.x)
            good = np.max(np.abs(u)) <= 2.0 + 1e-7 and (self.n == 0 or np.min(B @ u + C) >= -1e-7)
        if not good:
            self.fallbacks += 1
            u = clarabel_projection(target, B, C)
            if u is None: return np.zeros(3), False, int(result.info.iter)
            good = np.max(np.abs(u)) <= 2.0 + 1e-7 and (self.n == 0 or np.min(B @ u + C) >= -1e-7)
        return u, bool(good), int(result.info.iter)


def clarabel_projection(target, B, C):
    import clarabel
    import scipy.sparse as sp
    n = len(C)
    A = sp.csc_matrix(np.vstack([-B, np.eye(3), -np.eye(3)]))
    b = np.r_[C, [2.0] * 6]
    settings = clarabel.DefaultSettings(); settings.verbose = False
    settings.tol_gap_abs = 1e-10; settings.tol_gap_rel = 1e-10; settings.tol_feas = 1e-11
    settings.max_iter = 400
    answer = clarabel.DefaultSolver(sp.eye(3, format="csc"), -np.asarray(target), A, b,
                                    [clarabel.NonnegativeConeT(n + 6)], settings).solve()
    return np.asarray(answer.x) if str(answer.status) in ("Solved", "AlmostSolved") else None

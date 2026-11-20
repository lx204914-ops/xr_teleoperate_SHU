"""Persistent, all-obstacle hard QP. Infeasibility is a recorded failure."""
import time
import numpy as np
import cvxpy as cp


class SafetyQP:
    def __init__(self, count, cfg):
        self.cfg, self.count = cfg, count
        self.u = cp.Variable(3)
        self.nominal = cp.Parameter(3)
        self.previous = cp.Parameter(3, value=np.zeros(3))
        self.B = cp.Parameter((count, 3))
        self.c = cp.Parameter(count)
        limits = np.asarray(cfg.acceleration_limit)
        objective = .5*cp.sum_squares(self.u-self.nominal)
        objective += .5*cfg.qp_smoothing*cp.sum_squares(self.u-self.previous)
        self.problem = cp.Problem(cp.Minimize(objective),
                                  [self.B@self.u+self.c >= 0, self.u <= limits, self.u >= -limits])

    def solve(self, nominal, barriers, sampling_margin=None):
        margin = np.zeros(self.count) if sampling_margin is None else np.asarray(sampling_margin, dtype=float)
        if margin.shape != (self.count,) or not np.all(np.isfinite(margin)) or np.any(margin < 0):
            raise ValueError('Sampling margins must be finite, nonnegative and match the obstacle count.')
        self.nominal.value = nominal
        self.B.value = np.array([b['B'] for b in barriers])
        self.c.value = np.array([b['A']+self.cfg.cbf_gain*b['h'] for b in barriers]) - margin
        start = time.perf_counter()
        status = 'solver_error'
        for solver in ('CLARABEL', 'OSQP', 'SCS'):
            if solver not in cp.installed_solvers():
                continue
            try:
                self.problem.solve(solver=solver, warm_start=True, verbose=False)
            except cp.error.SolverError:
                continue
            status = self.problem.status
            if status in ('optimal', 'optimal_inaccurate') and self.u.value is not None:
                u = np.asarray(self.u.value).reshape(3)
                residual = self.B.value@u+self.c.value
                violation = max(float(np.max(np.abs(u)-self.cfg.acceleration_limit)),
                                -float(np.min(residual)))
                if np.all(np.isfinite(u)) and violation <= 1e-7:
                    self.previous.value = u.copy()
                    return u, status, float(np.min(residual)), time.perf_counter()-start
                status = 'residual_failure'
        return None, status, None, time.perf_counter()-start

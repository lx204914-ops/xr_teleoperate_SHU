import unittest
from dataclasses import replace
import numpy as np
from ra_dpcbf.core import Config, Boundary, evaluate, boundary_step, surface_q
from ra_dpcbf.qp import SafetyQP
from ra_dpcbf.theory import common_box_margin, single_box_margin


class MethodTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.x = np.array([.2, -.3, 1., .4, .1, -.2])
        self.o = np.array([3., .7, 2., -.2, .3, .1, .2, -.1, .05, .35])
        self.b = Boundary(1.2, .07, 1.5, .75, 2.25)

    def test_directional_derivative(self):
        rng = np.random.default_rng(73)
        for _ in range(40):
            u = rng.uniform(-1.5, 1.5, 3)
            b = evaluate(self.x, self.o, self.b, self.cfg)
            xd = np.r_[self.x[3:], u]
            od = np.r_[self.o[3:6], self.o[6:9], np.zeros(4)]
            eps = 1e-6
            plus = replace(self.b, radius=self.b.radius+eps*self.b.rate,
                           rate=self.b.rate+eps*self.b.acceleration(self.cfg))
            minus = replace(self.b, radius=self.b.radius-eps*self.b.rate,
                            rate=self.b.rate-eps*self.b.acceleration(self.cfg))
            dh = (evaluate(self.x+eps*xd, self.o+eps*od, plus, self.cfg)['h']
                  -evaluate(self.x-eps*xd, self.o-eps*od, minus, self.cfg)['h'])/(2*eps)
            self.assertAlmostEqual(dh, b['A']+b['B']@u, places=7)

    def test_boundary_consistency_and_surface(self):
        b = evaluate(self.x, self.o, self.b, self.cfg)
        self.assertAlmostEqual(b['h'], b['q']+b['mu_effective']+b['quadratic_coefficient']*b['Z'])
        self.assertAlmostEqual(b['h'], b['q']-surface_q(np.sqrt(b['Z']), 0, b['vertex'], b['quadratic_coefficient']))
        self.assertAlmostEqual(b['curvature'], 2*abs(b['quadratic_coefficient']))
        on = self.x.copy()
        on[:3] = self.o[:3]-self.b.radius*b['e']
        bb = evaluate(on, self.o, self.b, self.cfg)
        self.assertAlmostEqual(bb['h'], bb['q']-self.b.rate)
        self.assertAlmostEqual(np.linalg.norm(b['B'])**2, 1+4*self.cfg.beta**2*b['s']**2*b['Z'])

    def test_rotation_invariance(self):
        Q, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(3, 3)))
        x = np.r_[Q@self.x[:3], Q@self.x[3:]]
        o = np.r_[Q@self.o[:3], Q@self.o[3:6], Q@self.o[6:9], self.o[9]]
        a, b = evaluate(self.x, self.o, self.b, self.cfg), evaluate(x, o, self.b, self.cfg)
        self.assertAlmostEqual(a['h'], b['h'])
        self.assertAlmostEqual(a['A'], b['A'])
        np.testing.assert_allclose(Q@a['B'], b['B'], atol=1e-12)

    def test_no_reset_boundary_bounds(self):
        rng = np.random.default_rng(82)
        b = Boundary(.75, 0., .75, .75, 2.25)
        for _ in range(1000):
            b = replace(b, reference=float(rng.choice([.75, 2.25])))
            previous = b
            b = boundary_step(b, .02, self.cfg)
            self.assertGreaterEqual(b.radius, .75-1e-12)
            self.assertLessEqual(b.radius, 2.25+1e-12)
            self.assertLessEqual(abs(b.rate), self.cfg.omega*1.5+1e-12)
            self.assertLessEqual(abs(b.acceleration(self.cfg)), 2*self.cfg.omega**2*1.5+1e-12)
            self.assertLessEqual(abs(b.rate-previous.rate), 2*self.cfg.omega**2*1.5*.02+1e-12)

    def test_exact_flow_semigroup(self):
        full = boundary_step(self.b, .1, self.cfg)
        split = boundary_step(boundary_step(self.b, .04, self.cfg), .06, self.cfg)
        self.assertAlmostEqual(full.radius, split.radius, places=14)
        self.assertAlmostEqual(full.rate, split.rate, places=14)

    def test_hard_qp_and_joint_infeasibility(self):
        bar = evaluate(self.x, self.o, self.b, self.cfg)
        qp = SafetyQP(1, self.cfg)
        u, status, residual, _ = qp.solve(np.array([1., 0, 0]), [bar])
        self.assertIsNotNone(u)
        self.assertGreaterEqual(residual, -1e-6)
        self.assertGreaterEqual(single_box_margin(bar, self.cfg), 0)
        conflict = [dict(B=np.array([1., 0, 0]), A=-1., h=0.),
                    dict(B=np.array([-1., 0, 0]), A=-1., h=0.)]
        self.assertLess(common_box_margin(conflict, self.cfg), 0)
        self.assertIsNone(SafetyQP(2, self.cfg).solve(np.zeros(3), conflict)[0])

    def test_sampling_margin_reaches_generic_qp(self):
        barrier = dict(B=np.array([1., 0., 0.]), A=0., h=0.)
        u, _, residual, _ = SafetyQP(1, self.cfg).solve(np.zeros(3), [barrier], np.array([.4]))
        self.assertIsNotNone(u)
        self.assertGreaterEqual(u[0], .4-1e-7)
        self.assertGreaterEqual(residual, -1e-7)
        self.assertIsNone(SafetyQP(1, self.cfg).solve(np.zeros(3), [barrier], np.array([2.1]))[0])


if __name__ == '__main__':
    unittest.main()

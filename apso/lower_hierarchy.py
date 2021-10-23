import numpy as np
import sageopt as so
from sageopt import coniclifts as cl
from sageopt.relaxations.sage_sigs import hierarchy_e_k, relative_dual_sage_cone
from sageopt.relaxations import symbolic_correspondences as sym_corr
import sageopt.relaxations.sig_solution_recovery as sosr
import warnings


class DegreeTable(object):

    def __init__(self, A):
        self.A = A
        self.k, self.n = A.shape
        self.modulators = {1: so.Signomial(A, np.ones(self.k))}
        self.degs = dict()
        for a in self.A:
            self.degs[tuple(a)] = 1
        self.x = cl.Variable(shape=(self.k,), name='x')
        self.objective = cl.sum(self.x)
        pass

    def __getitem__(self, item):
        if item.size != self.n:
            raise ValueError()
        itup = tuple(item)
        if itup in self.degs:
            return self.degs[itup]
        else:
            prob = cl.Problem(cl.MIN, self.objective,
                              [self.x >= 0, item == self.A.T @ self.x],
                              integer_variables=[self.x])
            prob.solve(verbose=False)
            if prob.value is np.NaN:
                prob.solve(verbose=True)
            if prob.value < np.inf:
                val = int(prob.value)
                self.degs[itup] = max(val, 1)
                return val
            else:
                msg = """
                Encountered a monomial not inside the signomial ring induced by A.
                Returning degree np.inf.
                """
                warnings.warn(msg)
                return np.inf

    def update_modulators(self, d):
        cur_d = max(self.modulators.keys())
        for di in range(cur_d+1, d+1):
            self.modulators[di] = self.modulators[di-1] * self.modulators[1]
            for a in self.modulators[di].alpha:
                tupa = tuple(a)
                if tupa not in self.degs:
                    self.degs[tupa] = di
        pass

    def invsupp(self, g, d):
        deg_g = max([self[a] for a in g.alpha])
        if deg_g > d:
            msg = """
            The provided signomial has A-degree %s, but the %s-inverse
            support of g has been requested. The returned support set may
            be empty and will not contain the zero vector. This function
            may run for a long time. 
            """ % (str(deg_g), str(d))
            warnings.warn(msg)
            if deg_g == np.inf:
                return np.zeros(shape=(0, g.n))
        supp = [np.zeros(shape=(0, g.n))]
        self.update_modulators(d)
        Ad = self.modulators[d].alpha
        over_flag = False
        degkeys = self.degs.keys()
        for a in Ad:
            shift_galpha = g.alpha + a
            for ag in shift_galpha:
                tag = tuple(ag)
                over_flag = (tag not in degkeys) or (self.degs[tag] > d)
                if over_flag:
                    break
            if not over_flag:
                supp.append(a)
        supp = np.row_stack(supp)
        return supp


class FinderArray(object):

    def __init__(self, A):
        self.A = A
        self.row_map = dict()

    def find(self, a):
        tupa = tuple(a)
        if tupa in self.row_map:
            return self.row_map[tupa]
        else:
            shifted = self.A - a
            locs = np.where(np.all(np.abs(shifted) < 1e-8, axis=1))
            try:
                loc = locs[0][0]
                self.row_map[tupa] = loc
                return loc
            except IndexError:
                print('Row mismatch')
                return np.NaN


class SignomialProgram(object):

    NOT_IN_RING = """
    The signomial at index %s in this list does not
    belong to the current signomial ring.
    """

    BAD_DEGREE = """
    The signomial at index %s in this list cannot be moved
    into the Lagrangian for a SAGE relaxation of A-degree %s.
    The SAGE relaxation under construction will disregard this
    constraint.
    """

    def __init__(self, f, gts, eqs, X, A=None):
        self.f = f
        self.X = X
        self.gts = gts
        self.eqs = eqs
        if A is None:
            A = hierarchy_e_k(gts + eqs + [f], 1)
            self.total_deg = 1
            self.dt = DegreeTable(A)
        else:
            self.dt = DegreeTable(A)
            deg = max([self.dt[a] for a in f.alpha])
            for i, g in enumerate(gts):
                dg = max([self.dt[a] for a in g.alpha])
                if dg == np.inf:
                    raise ValueError(SignomialProgram.NOT_IN_RING % str(i))
                deg = max(deg, dg)
            for i, g in enumerate(eqs):
                dg = max([self.dt[a] for a in g.alpha])
                if dg == np.inf:
                    raise ValueError(SignomialProgram.NOT_IN_RING % str(i))
                deg = max(deg, dg)
            self.total_deg = deg
            if self.total_deg == np.inf:
                raise ValueError('Some signomial does not belong')

    def _make_lagrangian(self, d):
        if self.total_deg > d:
            raise ValueError()
        dt = self.dt
        dt.update_modulators(d)
        f, gts, eqs = self.f, self.gts, self.eqs
        gamma = cl.Variable(name='gamma')
        deg_f = max([dt[a] for a in f.alpha])
        f_mod = f - gamma
        if d > deg_f:
            modulator = dt.modulators[d - deg_f]
            f_mod = f_mod * modulator
        else:
            modulator = f.upcast_to_signomial(1)
        summands = [f_mod]
        ineq_dual_sigs = []
        for i, g in enumerate(gts):
            s_g_supp = dt.invsupp(g, d)
            if np.prod(s_g_supp.shape) == 0:
                msg = SignomialProgram.BAD_DEGREE % (str(i), str(d))
                warnings.warn(msg)
                continue
            s_g_coeff = cl.Variable(name='s_' + str(g), shape=(s_g_supp.shape[0],))
            s_g = so.Signomial(s_g_supp, s_g_coeff)
            summands.append(-g * s_g)
            ineq_dual_sigs.append((s_g, g))
        eq_dual_sigs = []
        for i, g in enumerate(eqs):
            z_g_supp = dt.invsupp(g, d)
            if np.prod(z_g_supp.shape) == 0:
                msg = SignomialProgram.BAD_DEGREE % (str(i), str(d))
                warnings.warn(msg)
                continue
            z_g_coeff = cl.Variable(name='z_' + str(g), shape=(z_g_supp.shape[0],))
            z_g = so.Signomial(z_g_supp, z_g_coeff)
            summands.append(-g * z_g)
            eq_dual_sigs.append((z_g, g))
        L = so.Signomial.sum(summands)
        return L, ineq_dual_sigs, eq_dual_sigs, gamma, modulator

    def primal_relaxation(self, d, sage=True, sage_mults=True):
        L, ineq_dual_sigs, eq_dual_sigs, gamma, modulator = self._make_lagrangian(d)
        cons = []
        for s_g, _ in ineq_dual_sigs:
            if not sage_mults:
                cons.append(s_g.c >= 0)
            else:
                name = '{%s}_{sage}' % str(s_g)
                con = cl.PrimalSageCone(s_g.c, s_g.alpha, self.X, name)
                cons.append(con)
        if sage:
            X = self.X
            con = cl.PrimalSageCone(L.c, L.alpha, X, name='sage_con')
        else:
            if self.X is not None:
                raise RuntimeError()
            con = L.c >= 0
        cons.append(con)
        prob = cl.Problem(cl.MAX, gamma, cons)
        metadata = {'f': self.f, 'gts': self.gts, 'eqs': self.eqs, 'X': self.X,
                    'd': d, 'A': self.dt.A, 'lagrangian': L, 'modulator': modulator,
                    'ineq_dual_sigs': ineq_dual_sigs, 'eq_dual_sigs': eq_dual_sigs}
        prob.metadata = metadata
        cl.clear_variable_indices()
        return prob

    @staticmethod
    def _fast_moment_reduction_array(s_h, h, lag_A):
        c_h = np.zeros((s_h.m, lag_A.A.shape[0]))
        for i, alpha_i in enumerate(s_h.alpha):
            shift = h.alpha + alpha_i
            for j, alpha_j in enumerate(shift):
                loc = lag_A.find(alpha_j)
                if loc is not np.NaN:
                    c_h[i, loc] = h.c[j]
        return c_h

    def dual_relaxation(self, d, sage_mults=True):
        lagrangian, ineq_dual_sigs, eq_dual_sigs, _, modulator = self._make_lagrangian(d)
        lag_A = FinderArray(lagrangian.alpha)
        v = cl.Variable(shape=(lagrangian.m, 1), name='v')
        con = relative_dual_sage_cone(lagrangian, v, name='Lagrangian SAGE dual constraint', X=self.X)
        constraints = [con]
        for i, (s_h, h) in enumerate(ineq_dual_sigs):
            c_h = self._fast_moment_reduction_array(s_h, h, lag_A)
            con_suffix = ' sig ineq %s' % str(i)
            if sage_mults:
                c_h_v = c_h @ v
                con = relative_dual_sage_cone(s_h, c_h_v, name='Dual sage' + con_suffix, X=self.X)
            else:
                con = c_h @ v >= 0
                con.name += con_suffix
            constraints.append(con)
        for i, (s_h, h) in enumerate(eq_dual_sigs):
            c_h = self._fast_moment_reduction_array(s_h, h, lag_A)
            con = c_h @ v == 0
            con.name += ' sig equality %s' % str(i)
            constraints.append(con)
        a = sym_corr.relative_coeff_vector(modulator, lagrangian.alpha)
        con = a.T @ v == 1
        con.name += ' dehomogenize'
        constraints.append(con)
        f_mod = self.f * modulator
        obj_vec = sym_corr.relative_coeff_vector(f_mod, lagrangian.alpha)
        obj = obj_vec.T @ v
        prob = cl.Problem(cl.MIN, obj, constraints)
        cl.clear_variable_indices()
        metadata = {'f': self.f, 'gts': self.gts, 'eqs': self.eqs, 'X': self.X,
                    'd': d, 'A': self.dt.A, 'lagrangian': lagrangian, 'modulator': modulator,
                    'ineq_dual_sigs': ineq_dual_sigs, 'eq_dual_sigs': eq_dual_sigs}
        prob.metadata = metadata
        return prob


def sig_solrec(prob, ineq_tol=1e-8, eq_tol=1e-6, skip_ls=False):
    con = prob.constraints[0]
    if not con.name == 'Lagrangian SAGE dual constraint':  # pragma: no cover
        raise RuntimeError('Unexpected first constraint in dual SAGE relaxation.')
    metadata = prob.metadata
    f = metadata['f']
    # Recover any constraints present in "prob"
    lag_gts, lag_eqs = [], []
    if 'gts' in metadata:
        # only happens in "constrained_sage_dual".
        lag_gts = metadata['gts']
        lag_eqs = metadata['eqs']
    if con.X is None:
        X_gts, X_eqs = [], []
    else:
        X_gts, X_eqs = con.X.gts, con.X.eqs
    gts = lag_gts + X_gts
    eqs = lag_eqs + X_eqs
    # Search for solutions which meet the feasibility criteria
    v = con.v.value
    v[v < 0] = 0
    if np.any(np.isnan(v)):
        return None
    lag_alpha = FinderArray(prob.metadata['lagrangian'].alpha)
    A = prob.metadata['A']
    target = so.Signomial(A, np.ones(A.shape[0]))
    modulator = prob.metadata['modulator']
    M = SignomialProgram._fast_moment_reduction_array(target, modulator, lag_alpha)
    if skip_ls:
        sols0 = []
    else:
        sols0 = sosr._least_squares_solution_recovery(A, con, v, M, gts, eqs, ineq_tol, eq_tol)
    sols1 = sosr._dual_age_cone_solution_recovery(con, v, M, gts, eqs, ineq_tol, eq_tol)
    sols = sols0 + sols1
    sols.sort(key=lambda mu: f(mu))
    return sols


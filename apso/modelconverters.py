import pyomo.repn.plugins.gams_writer as GamsWriter
import pyomo.environ as pyo
import numpy as np
import sageopt as so


def eval_geoform_sig(f, x):
    fx = 0.0
    for i in range(f.m):
        expr = 1.0
        for j in range(f.n):
            expr *= (x[j]**f.alpha[i, j])
        expr *= f.c[i]
        fx += expr
    return fx


def eval_expform_sig(f, x):
    fx = 0.0
    for i in range(f.m):
        expr = f.c[i] * pyo.exp(pyo.summation(f.alpha[i, :], x))
        fx += expr
    return fx


def clear_den(f):
    a = np.min(f.alpha, axis=0)
    a[a > 0] = 0
    c = f.c.copy()
    alpha = f.alpha - a
    f_mod = so.Signomial(alpha, c)
    return f_mod


def pyomo_model(f, gts, eqs, form='exp', keep_dens=False):
    m = pyo.ConcreteModel()
    m.gts = pyo.ConstraintList()
    m.eqs = pyo.ConstraintList()
    transform = (lambda g: g) if keep_dens else clear_den
    if form == 'geo':
        m.t = pyo.Var(np.arange(f.n, dtype=int), within=pyo.PositiveReals)
        for g in gts:
            g_mod = transform(g)
            m.gts.add(eval_geoform_sig(g_mod, m.t) >= 0)
        for g in eqs:
            g_mod = transform(g)
            m.eqs.add(eval_geoform_sig(g_mod, m.t) == 0)
        if np.any(f.alpha < 0) and not keep_dens:
            m.w = pyo.Var(within=pyo.Reals)
            m.f = pyo.Objective(expr=m.w)
            a = np.min(f.alpha, axis=0)
            a[a > 0] = 0
            f_mod = so.Signomial(f.alpha - a, f.c)
            lower = eval_geoform_sig(f_mod, m.t)
            upper = m.w * pyo.prod([m.t[i]**(-a[i]) for i in range(f.n)])
            m.gts.add(lower <= upper)
        else:
            m.f = pyo.Objective(expr=eval_geoform_sig(f, m.t))
    else:
        m.x = pyo.Var(np.arange(f.n, dtype=int), within=pyo.Reals)
        for g in gts:
            m.gts.add(eval_expform_sig(g, m.x) >= 0)
        for g in eqs:
            m.eqs.add(eval_expform_sig(g, m.x) == 0)
        m.f = pyo.Objective(expr=eval_expform_sig(f, m.x))
    return m


def pyomo2gams(m, name):
    pwg = GamsWriter.ProblemWriter_gams()
    pwg(m, '%s.gams' % name, lambda x: True,
        io_options={'solprint': 'on',
                    'symbolic_solver_labels': True})
    pass


def signomial_substitution(f, monovec):
    temp1 = np.power(monovec, f.alpha)
    # ^ broadcast; each row of f.alpha elementwise exponentiates monovec
    temp2 = np.prod(temp1, axis=1)
    # ^ reduce: each row of the output matrix is collapsed to form a monomial
    temp3 = f.c * temp2
    # ^ elementwise scaling
    val = so.Signomial.sum(temp3.tolist())
    return val

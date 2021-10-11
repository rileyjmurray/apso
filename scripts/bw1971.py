import numpy as np
import sageopt as so
import apso.modelconverters as mc
import apso.lower_hierarchy as lh
from scipy.optimize import fmin_cobyla


def write_gams_eqform(form='exp'):
    x_geo_bw_init = np.array([1e3, 1e1, 1e5, 1e2, 1e5, 1e3, 1e-1, 1e1])
    f, gts, eqs, X = blau_wilde(auto_scale=x_geo_bw_init)
    m = mc.pyomo_model(f, gts, eqs, form, keep_dens=True)
    import pyomo.repn.plugins.gams_writer as GamsWriter
    pwg = GamsWriter.ProblemWriter_gams()
    pwg(m, 'bw1971/apr6_%s.gams' % form, lambda x: True,
        io_options={'solprint': 'on',
                    'symbolic_solver_labels': True,
                    'sysOut': 'on'})


def blau_wilde(auto_scale=False):
    n = 8
    t = so.standard_sig_monomials(n)
    if isinstance(auto_scale, np.ndarray) or auto_scale:
        if isinstance(auto_scale, bool):
            x_geo_bw = np.array([5160, 6.65, 171000, 749, 86000, 187, 0.124, 29.1])
        elif isinstance(auto_scale, np.ndarray) and auto_scale.size == n:
            x_geo_bw = auto_scale
        else:
            raise ValueError()
        for i in range(n):
            t[i] = x_geo_bw[i] * t[i]

    f1 = 2.0425*(t[0]**0.782) + 52.25*t[1] + 192.85*(t[1]**0.9) + 5.25*(t[1]**3) + 61.465*(t[5]**0.467)
    f2 = 0.01748 * (t[2]**1.33)/(t[3]**0.8) + 100.7 * (t[3] ** 0.546) + 3.66e-10*(t[2] ** 2.85 * t[3] ** -1.7)
    f3 = 0.00945*t[4] + 1.06e-10*(t[4]**2.8)/(t[3]**1.8) + 116*t[5] - 205*t[5]*t[6] - 278*(t[1]**3)*t[6]
    f = f1 + f2 + f3

    # build constraints as gs[i](x) <= 0
    gs = [129.4/(t[1]**3) + 105/t[5] - 1,
          1.03e5*(t[1]**3)*t[6]/(t[2]*t[7]) + 1.2e6/(t[2]*t[7]) - 1,
          4.68*(t[1]**3)/t[0] + 61.3*(t[1]**2)/t[0] + 160.5*t[1]/t[0] - 1,
          1.79*t[6] + 3.02*(t[1]**3)*t[6]/t[5] + 35.7/t[5] - 1,
          1.22e-3*t[2]*t[7]/((t[3]**0.2)*(t[4]**0.8)) + 1.67e-3*t[7]*(t[2]**0.4)/(t[3]**0.43)
            + 3.6e-5*t[2]*t[7]/t[3] + 2e-3*t[2]*t[7]/t[4] + 4e-3*t[7] - 1]
    # convert to gs[i](x) >= 0
    gs = [-g for g in gs]
    X = so.infer_domain(f, gs, [])

    if isinstance(auto_scale, np.ndarray) or auto_scale:
        f = f / 1e4

    # return gs as equality constraints (per our manuscript)
    return f, [], gs, X


def run_sage_relaxations():
    x_geo_bw_init = np.array([1e3, 1e1, 1e5, 1e2, 1e5, 1e3, 1e-1, 1e1])
    f, gts, eqs, X = blau_wilde(auto_scale=x_geo_bw_init)
    sp = lh.SignomialProgram(f, [], gts, X)
    dual1 = sp.dual_relaxation(d=1, sage_mults=True)
    dual1.solve()
    dual2 = sp.dual_relaxation(d=2, sage_mults=True)
    dual2.solve()
    """
    Optimizer terminated. Time: 0.13    
      Primal.  obj: 1.6377324226e+00    nrm: 1e+01    Viol.  con: 1e-08    var: 2e-09    cones: 0e+00  
      Dual.    obj: 1.6377324346e+00    nrm: 5e+01    Viol.  con: 0e+00    var: 7e-11    cones: 0e+00  
    ('solved', 1.6377324225847878) --> 16377.324225847879

    Optimizer terminated. Time: 24.37   
      Primal.  obj: 1.7462726685e+00    nrm: 3e+03    Viol.  con: 3e-06    var: 1e-08    cones: 0e+00  
      Dual.    obj: 1.7462726690e+00    nrm: 1e+01    Viol.  con: 0e+00    var: 4e-09    cones: 0e+00  
    ('solved', 1.7462726685073702) --> 17462.726685073703
    """
    return sp, dual1, dual2


if __name__ == '__main__':
    # sp, d1, d2 = config_2()
    write_gams_eqform('geo')

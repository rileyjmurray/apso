import matplotlib.pyplot as plt
import numpy as np
import sageopt as so
from apso import lower_hierarchy as lh

SMALL_SIZE = 16
MEDIUM_SIZE = 16
BIGGER_SIZE = 26

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def get_model():
    t = so.standard_sig_monomials(9)
    k = {1: t[0],
         2: t[1],
         3: t[2],
         4: t[3],
         5: t[4],
         7: t[5],
         8: t[6]}
    x = {2: t[7],
         3: t[8]}
    K = {
        (0, 0): (1+k[2])*(1+k[4])*(1+k[5])*(1+k[8]),
        (1, 0): (1+k[8])*(k[1] + k[3] + k[1]*k[4] + k[1]*k[5] + k[2]*k[3] + k[1]*k[4]*k[5]),
        (0, 1): (1+k[2])*(k[3] + 4*k[7] + k[3]*k[5] + k[3]*k[8] + 4*k[4]*k[7] + k[3]*k[5]*k[8]),
        (2, 0): k[1]*k[3]*(1+k[8]),
        (1, 1): k[1]*(k[3] + 4*k[7] + k[3]*k[8] + 4*k[4]*k[7] - k[3]*k[5] - k[3]*k[5]*k[8]),
        (0, 2): 4*k[3]*k[7]*(1+k[2]),
        (1, 2): 4*k[1]*k[3]*k[7]
    }
    fk = K[(0, 0)] + K[(1, 0)]*x[2] + K[(0, 1)]*x[3] + K[(2, 0)]*x[2]**2 + K[(1, 1)]*x[2]*x[3]
    fk += K[(0, 2)]*x[3]**2 + K[(1, 2)]*x[2]*x[3]**2
    return fk, k, x


def box_bounds_2(fk: so.Signomial, k: dict, lowbox: float, hibox: float, k5: float):
    box_ks = [hibox - ki for idx, ki in k.items() if idx != 5]
    box_ks += [ki - lowbox for idx, ki in k.items() if idx != 5]
    logT = so.infer_domain(fk, box_ks, [k[5] - k5], check_feas=False)
    return logT


if __name__ == '__main__':
    fk, k, x = get_model()
    X = box_bounds_2(fk, k, 1/2.41, 2.41, 7.06)
    A_nat = np.row_stack((np.eye(fk.n), np.zeros(fk.n)))
    A_mid = np.row_stack((A_nat, fk.alpha[fk.c < 0, :]))
    sp_nat = lh.SignomialProgram(fk, [], [], X, A_nat)
    sp_mid = lh.SignomialProgram(fk, [], [], X, A_mid)
    sp_nav = lh.SignomialProgram(fk, [], [], X, fk.alpha)
    # Natural rings
    nat6 = sp_nat.dual_relaxation(d=6)
    nat7 = sp_nat.dual_relaxation(d=7)
    nat8 = sp_nat.dual_relaxation(d=8)
    # Mid rings
    mid6 = sp_mid.dual_relaxation(d=6)
    mid7 = sp_mid.dual_relaxation(d=7)
    # Naive rings
    nav1 = sp_nav.dual_relaxation(d=1)
    nav2 = sp_nav.dual_relaxation(d=2)
    """
    nat6.solve(verbose=False)
    ('solved', 18.15964213531008)
    nat7.solve(verbose=False)
    ('solved', 18.718883494594238)
    nat8.solve(verbose=False)
    ('solved', 19.73755180041542)
    mid6.solve(verbose=False)
    ('solved', 18.15964213531008)
    mid7.solve(verbose=False)
    ('solved', 22.832163944741463)
    nav1.solve(verbose=False)
    ('solved', 18.15964213531008)
    nav2.solve(verbose=False)
    ('solved', 22.832165711613175)
    """
    """
    nat6.timings['MOSEK']
    {'apply': 0.006966590881347656, 'solve_via_data': 0.034491539001464844, 'total': 0.041739702224731445}
    nat7.timings['MOSEK']
    {'apply': 0.06282782554626465, 'solve_via_data': 1.0541613101959229, 'total': 1.1176090240478516}
    nat8.timings['MOSEK']
    {'apply': 0.6313450336456299, 'solve_via_data': 49.20002031326294, 'total': 49.83453035354614}
    mid6.timings['MOSEK']
    {'apply': 0.0052187442779541016, 'solve_via_data': 0.030153989791870117, 'total': 0.03563117980957031}
    mid7.timings['MOSEK']
    {'apply': 0.047302961349487305, 'solve_via_data': 1.1123239994049072, 'total': 1.160271406173706}
    nav1.timings['MOSEK']
    {'apply': 0.005788564682006836, 'solve_via_data': 0.032196998596191406, 'total': 0.038205623626708984}
    nav2.timings['MOSEK']
    {'apply': 0.1643667221069336, 'solve_via_data': 3.464855432510376, 'total': 3.630207061767578}

    """

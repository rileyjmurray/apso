import warnings

import matplotlib.pyplot as plt
import numpy as np
import sageopt as so
import pickle
import dask
from apso import lower_hierarchy as lh
from apso import util

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


def box_bounds_1(fk: so.Signomial, k: dict, pow: float, k5: float):
    box_ks = [10**pow - ki for idx, ki in k.items() if idx != 5]
    box_ks += [ki - 10**-pow for idx, ki in k.items() if idx != 5]
    logT = so.infer_domain(fk, box_ks, [k[5] - k5], check_feas=False)
    return logT


def make_prob_1(fk: so.Signomial, k: dict, delta: float, k5: float, ell: int):
    logT = box_bounds_1(fk, k, delta, k5)
    prob = so.sig_relaxation(fk, logT, ell=ell)
    return prob


def upper_lower(fk: so.Signomial, k: dict, delta: float, k5: float, ell: int):
    prob = make_prob_1(fk, k, delta, k5, ell)
    prob.solve()
    logT = prob.metadata['X']
    upper = np.NaN
    if prob.status == so.coniclifts.SOLVED and prob.value > -np.inf:
        sols = so.sig_solrec(prob)
        val = fk(sols[0])
        rhobeg = min(1, abs(val - prob.value)/(1+abs(val)))
        logt_ref = so.local_refine(fk, logT.gts, logT.eqs, sols[0], rhobeg=rhobeg)
        upper = fk(logt_ref)
        if upper > val:
            warnings.warn('COBYLA made things worse')
            upper = val
    return upper, prob.value


def box_bounds_2(fk: so.Signomial, k: dict, lowbox: float, hibox: float, k5: float):
    box_ks = [hibox - ki for idx, ki in k.items() if idx != 5]
    box_ks += [ki - lowbox for idx, ki in k.items() if idx != 5]
    logT = so.infer_domain(fk, box_ks, [k[5] - k5], check_feas=False)
    return logT


def make_prob_2(fk: so.Signomial, k: dict, lowbox: float, hibox: float, k5: float, ell: int):
    logT = box_bounds_2(fk, k, lowbox, hibox, k5)
    prob = so.sig_relaxation(fk, logT, ell=ell)
    return prob


def fk_str():
    # This function horizontally concatenates two large LaTeX tables.
    # The data for the tables was precomputed
    s1 = """
    $\cdot$ & 1 & $\cdot$ & 1 & 1 & $\cdot$ & 1 & $\cdot$ & $\cdot$ &   1 & 
    $\cdot$ & $\cdot$ & $\cdot$ & 1 & 1 & $\cdot$ & 1 & $\cdot$ & $\cdot$ &   1 & 
    $\cdot$ & 1 & $\cdot$ & $\cdot$ & 1 & $\cdot$ & 1 & $\cdot$ & $\cdot$ &   1 & 
    $\cdot$ & $\cdot$ & $\cdot$ & $\cdot$ & 1 & $\cdot$ & 1 & $\cdot$ & $\cdot$ &   1 & 
    $\cdot$ & 1 & $\cdot$ & 1 & $\cdot$ & $\cdot$ & 1 & $\cdot$ & $\cdot$ &   1 & 
    $\cdot$ & $\cdot$ & $\cdot$ & 1 & $\cdot$ & $\cdot$ & 1 & $\cdot$ & $\cdot$ &   1 & 
    $\cdot$ & 1 & $\cdot$ & $\cdot$ & $\cdot$ & $\cdot$ & 1 & $\cdot$ & $\cdot$ &   1 & 
    $\cdot$ & $\cdot$ & $\cdot$ & $\cdot$ & $\cdot$ & $\cdot$ & 1 & $\cdot$ & $\cdot$ &   1 & 
    $\cdot$ & 1 & $\cdot$ & 1 & 1 & $\cdot$ & $\cdot$ & $\cdot$ & $\cdot$ &   1 & 
    $\cdot$ & $\cdot$ & $\cdot$ & 1 & 1 & $\cdot$ & $\cdot$ & $\cdot$ & $\cdot$ &   1 & 
    $\cdot$ & 1 & $\cdot$ & $\cdot$ & 1 & $\cdot$ & $\cdot$ & $\cdot$ & $\cdot$ &   1 & 
    $\cdot$ & $\cdot$ & $\cdot$ & $\cdot$ & 1 & $\cdot$ & $\cdot$ & $\cdot$ & $\cdot$ &   1 & 
    $\cdot$ & 1 & $\cdot$ & 1 & $\cdot$ & $\cdot$ & $\cdot$ & $\cdot$ & $\cdot$ &   1 & 
    $\cdot$ & $\cdot$ & $\cdot$ & 1 & $\cdot$ & $\cdot$ & $\cdot$ & $\cdot$ & $\cdot$ &   1 & 
    $\cdot$ & 1 & $\cdot$ & $\cdot$ & $\cdot$ & $\cdot$ & $\cdot$ & $\cdot$ & $\cdot$ &   1 & 
    $\cdot$ & $\cdot$ & $\cdot$ & $\cdot$ & $\cdot$ & $\cdot$ & $\cdot$ & $\cdot$ & $\cdot$ &   1 & 
    1 & $\cdot$ & $\cdot$ & $\cdot$ & $\cdot$ & $\cdot$ & 1 & 1 & $\cdot$ &   1 & 
    1 & $\cdot$ & $\cdot$ & $\cdot$ & $\cdot$ & $\cdot$ & $\cdot$ & 1 & $\cdot$ &   1 & 
    $\cdot$ & $\cdot$ & 1 & $\cdot$ & $\cdot$ & $\cdot$ & 1 & 1 & $\cdot$ &   1 & 
    $\cdot$ & $\cdot$ & 1 & $\cdot$ & $\cdot$ & $\cdot$ & $\cdot$ & 1 & $\cdot$ &   1 & 
    1 & $\cdot$ & $\cdot$ & 1 & $\cdot$ & $\cdot$ & 1 & 1 & $\cdot$ &   1 & 
    1 & $\cdot$ & $\cdot$ & 1 & $\cdot$ & $\cdot$ & $\cdot$ & 1 & $\cdot$ &   1 & 
    1 & $\cdot$ & $\cdot$ & $\cdot$ & 1 & $\cdot$ & 1 & 1 & $\cdot$ &   1 & 
    1 & $\cdot$ & $\cdot$ & $\cdot$ & 1 & $\cdot$ & $\cdot$ & 1 & $\cdot$ &   1 & 
    $\cdot$ & 1 & 1 & $\cdot$ & $\cdot$ & $\cdot$ & 1 & 1 & $\cdot$ &   1 & 
    $\cdot$ & 1 & 1 & $\cdot$ & $\cdot$ & $\cdot$ & $\cdot$ & 1 & $\cdot$ &   1 & 
    """
    s2 = """
    1 & $\cdot$ & $\cdot$ & 1 & 1 & $\cdot$ & 1 & 1 & $\cdot$ &   1 \\\\
    1 & $\cdot$ & $\cdot$ & 1 & 1 & $\cdot$ & $\cdot$ & 1 & $\cdot$ &   1 \\\\
    $\cdot$ & 1 & 1 & $\cdot$ & $\cdot$ & $\cdot$ & $\cdot$ & $\cdot$ & 1 &   1 \\\\
    $\cdot$ & $\cdot$ & 1 & $\cdot$ & $\cdot$ & $\cdot$ & $\cdot$ & $\cdot$ & 1 &   1 \\\\
    $\cdot$ & 1 & $\cdot$ & $\cdot$ & $\cdot$ & 1 & $\cdot$ & $\cdot$ & 1 &   4 \\\\
    $\cdot$ & $\cdot$ & $\cdot$ & $\cdot$ & $\cdot$ & 1 & $\cdot$ & $\cdot$ & 1 &   4 \\\\
    $\cdot$ & 1 & 1 & $\cdot$ & 1 & $\cdot$ & $\cdot$ & $\cdot$ & 1 &   1 \\\\
    $\cdot$ & $\cdot$ & 1 & $\cdot$ & 1 & $\cdot$ & $\cdot$ & $\cdot$ & 1 &   1 \\\\
    $\cdot$ & 1 & 1 & $\cdot$ & $\cdot$ & $\cdot$ & 1 & $\cdot$ & 1 &   1 \\\\
    $\cdot$ & $\cdot$ & 1 & $\cdot$ & $\cdot$ & $\cdot$ & 1 & $\cdot$ & 1 &   1 \\\\
    $\cdot$ & 1 & $\cdot$ & 1 & $\cdot$ & 1 & $\cdot$ & $\cdot$ & 1 &   4 \\\\
    $\cdot$ & $\cdot$ & $\cdot$ & 1 & $\cdot$ & 1 & $\cdot$ & $\cdot$ & 1 &   4 \\\\
    $\cdot$ & 1 & 1 & $\cdot$ & 1 & $\cdot$ & 1 & $\cdot$ & 1 &   1 \\\\
    $\cdot$ & $\cdot$ & 1 & $\cdot$ & 1 & $\cdot$ & 1 & $\cdot$ & 1 &   1 \\\\
    1 & $\cdot$ & 1 & $\cdot$ & $\cdot$ & $\cdot$ & 1 & 2 & $\cdot$ &   1 \\\\
    1 & $\cdot$ & 1 & $\cdot$ & $\cdot$ & $\cdot$ & $\cdot$ & 2 & $\cdot$ &   1 \\\\
    1 & $\cdot$ & 1 & $\cdot$ & $\cdot$ & $\cdot$ & $\cdot$ & 1 & 1 &   1 \\\\
    1 & $\cdot$ & $\cdot$ & $\cdot$ & $\cdot$ & 1 & $\cdot$ & 1 & 1 &   4 \\\\
    1 & $\cdot$ & 1 & $\cdot$ & $\cdot$ & $\cdot$ & 1 & 1 & 1 &   1 \\\\
    1 & $\cdot$ & $\cdot$ & 1 & $\cdot$ & 1 & $\cdot$ & 1 & 1 &   4 \\\\
    1 & $\cdot$ & 1 & $\cdot$ & 1 & $\cdot$ & $\cdot$ & 1 & 1 &   -1 \\\\
    1 & $\cdot$ & 1 & $\cdot$ & 1 & $\cdot$ & 1 & 1 & 1 &   -1 \\\\
    $\cdot$ & 1 & 1 & $\cdot$ & $\cdot$ & 1 & $\cdot$ & $\cdot$ & 2 &   4 \\\\
    $\cdot$ & $\cdot$ & 1 & $\cdot$ & $\cdot$ & 1 & $\cdot$ & $\cdot$ & 2 &   4 \\\\
    1 & $\cdot$ & 1 & $\cdot$ & $\cdot$ & 1 & $\cdot$ & 1 & 2 &   4
    """
    row_strs = []
    s1_lines = s1.split('\n')[1:-1]
    s2_lines = s2.split('\n')[1:-1]
    for i in range(25):
        row = s1_lines[i] + s2_lines[i]
        row_strs.append(row)
    row = s1_lines[-1]
    row += (' &'*9)
    row_strs.append(row)
    table_content = '\n'.join(row_strs)
    return table_content


if __name__ == '__main__':
    compute_all = True
    fk, k, x = get_model()
    n_kalt = 50
    n_k5 = 50

    if compute_all:
        pows = np.log10(np.linspace(10**0.2, 10**0.75, n_kalt))
        k5s = np.linspace(1, 10, n_k5)
        k5ss, powss = np.meshgrid(k5s, pows)

        tasks = []
        for i in range(n_kalt):
            for j in range(n_k5):
                po, k5 = powss[i, j], k5ss[i, j]
                task = dask.delayed(upper_lower)(fk, k, po, k5, ell=0)
                tasks.append(task)
        tasks = dask.compute(*tasks, scheduler='processes')

        lowers = np.zeros(shape=(n_kalt, n_k5))
        uppers = np.zeros(shape=(n_kalt, n_k5))
        count = 0
        for i in range(n_kalt):
            for j in range(n_k5):
                ul = tasks[count]
                uppers[i, j] = ul[0]
                lowers[i, j] = ul[1]
                count += 1
        del task
        del tasks

        d = util.meaningful_locals(locals())
        fl = open('crn/sageres.pkl', 'wb')
        pickle.dump(d, fl)
        fl.close()
        print('Done with run!')

        horzax = k5ss.ravel()
        vertax = (10**powss).ravel()
        vals = (lowers > 0).ravel()
        colors = np.empty(shape=vals.shape, dtype=str)
        colors[vals] = 'b'
        colors[~vals] = 'r'
        ax = plt.gca()
        ax.scatter(vertax, horzax, c=colors)
        plt.show()
    else:
        fl = open('crn/sageres.pkl', 'rb')  # everything from ex2.
        d = pickle.load(fl)
        fl.close()
        for key in d.keys():
            exec(key + " = d['%s']" % key)
            # ^ This can't actually be used in a function call.
            # Can only be used in the REPL. Apparently this has
            # something to do with compilation into bytecode precluding
            # the definition of new variables after the fact (unless
            # you're in the REPL).
    gaps = uppers - lowers
    am = np.unravel_index(np.argmax(gaps), gaps.shape)
    k5, po = k5ss[am], powss[am]
    """
    k5
    7.061224489795919
    10**po
    2.409080959694252
    10**-po
    0.4150960539437059
    """
    prob0 = make_prob_2(fk, k, 0.415, 2.41, 7.06, ell=0)  # just for X
    X = prob0.metadata['X']
    A_nat = np.row_stack((np.eye(fk.n), np.zeros(fk.n)))
    A_mid = np.row_stack((A_nat, fk.alpha[fk.c < 0, :]))
    sp_nat = lh.SignomialProgram(fk, [], [], X, A_nat)
    sp_mid = lh.SignomialProgram(fk, [], [], X, A_mid)
    sp_nav = lh.SignomialProgram(fk, [], [], X, fk.alpha)
    """
    A_nat = np.row_stack((np.eye(fk.n), np.zeros(fk.n)))
    sp_nat = lh.SignomialProgram(fk, [], [], X, A_nat)
    nat6 = sp_nat.dual_relaxation(d=6)
    nat7 = sp_nat.dual_relaxation(d=7)
    nat8 = sp_nat.dual_relaxation(d=8)
    """
    dual7 = sp_mid.dual_relaxation(7)
    dual7.solve(verbose=True)
    """

    """

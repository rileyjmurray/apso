import numpy as np
import sageopt as so
import apso.modelconverters as mc
import apso.lower_hierarchy as lh
from scipy.optimize import fmin_cobyla
import pickle


def rm1978_p23(all_lag=False):
    n = 5
    y = so.standard_sig_monomials(n)
    # HSC2014 (example 5) claims OPT \approx 10122.
    # RM1978 (problem 23) claims OPT \approx 10127.13.
    # MCW2019 reports a lower bound of 9171.
    """
    RM1978 Problem 23
        Cites COLVILLE, A. R., A Comparative Study of Nonlinear Programming Codes, IBM
        New York Scientific Center, Report No. 320-2949, 1968.
            I can find an article by Colville with this name published in 1971.
            That article doesn't give problem formulations explicitly and
            doesn't describe statistics for a problem that matches this one.
            Presumably the cited report is a different more comprehensive document
            and contains the formulation explicitly. Since that's proving hard
            to find, I'm going to cite RM1978 as the source.
    HSC2014 Example 5.
        Cites 21,22,24,33 -- all from the 2000's.
    See also the paper on mutation of test problems in signomial geometric programming.
    """
    f = 5.3578 * y[2] ** 2 + 0.8357 * y[0] * y[4] + 37.2392 * y[0]
    gts = [0.02584 * y[2] * y[4] - 0.06663 * y[1] * y[4] - 0.0734 * y[0] * y[3] - 1000,
           -0.33085 * y[2] * y[4] + 0.853007 * y[1] * y[4] + 0.09395 * y[0] * y[3] - 1000,
           1330.3294/(y[1]*y[4]) - .42*y[0]/y[4] - .30586*(y[2]**2)/(y[1]*y[4]) - 1,
           2275.1327/(y[2]*y[4]) - 0.2668*y[0]/y[4] - 0.40584*y[3]/y[4] - 1,
           0.24186 * y[1] * y[4] + 0.10159 * y[0] * y[1] + 0.07379 * y[2] ** 2 - 1000,
           0.29955 * y[2] * y[4] + 0.07992 * y[0] * y[2] + 0.12157 * y[2] * y[3] - 1000]

    gts = [-g for g in gts]
    siggts = gts[:-2]
    gpgts = gts[-2:] + [102 - y[0], y[0] - 78,
                        45 - y[1], y[1] - 33,
                        45 - y[2], y[2] - 27,
                        45 - y[3], y[3] - 27,
                        45 - y[4], y[4] - 27]
    eqs = []
    X = so.infer_domain(f, gpgts, [])
    if all_lag:
        gts = siggts + gpgts
    else:
        gts = siggts
    return f, gts, eqs, X


def support_info(dual):
    meta = dual.metadata
    print('\nSupport sizes for Lagrange multipliers')
    for i, (s_g, g) in enumerate(meta['ineq_dual_sigs']):
        print('\t Constraint %s: %s' % (i, s_g.m))
    print('\nThe exponents for the signomial ring')
    print(sp.dt.A)


def ring_support_sizes():
    sizes = []
    f, gts, eqs, X = rm1978_p23(True)
    fsupp = {tuple(a) for a in f.alpha}
    gsupp0 = {tuple(a) for g in gts[:4] for a in g.alpha}
    runsupp = gsupp0.union(fsupp)
    print('Support size for objective and nonconvex constraints')
    sizes.append(len(runsupp))
    print(sizes[-1])
    gsupp1 = {tuple(a) for g in gts[4:6] for a in g.alpha}
    runsupp = runsupp.union(gsupp1)
    print('Support size for everything except bound constraints')
    sizes.append(len(runsupp))
    print(sizes[-1])
    gsupp2 = {tuple(a) for g in gts[6:] for a in g.alpha}
    runsupp = runsupp.union(gsupp2)
    print('Support size for full Lagrangian')
    sizes.append(len(runsupp))
    print(sizes[-1])
    print('Support size for lagrangian omitting non-bound gp constraints')
    supp = fsupp.union(gsupp0).union(gsupp2)
    sizes.append(len(supp))
    print(sizes[-1])
    return sizes


def run_config(f, gts, eqs, X, lag, A_strat, d_min, d_max, verb):
    if lag == 'no box':
        gts = gts[:-10]
    elif lag == 'minimal':
        gts = gts[:4]
    elif lag != 'full':
        raise RuntimeError()
    if A_strat == 'default':
        A = None
    elif A_strat == 'custom':
        A = np.row_stack([np.eye(f.n), np.zeros(f.n)])
    else:
        raise RuntimeError()
    sp = lh.SignomialProgram(f, gts, eqs, X, A=A)
    vals = []
    times = []
    ds = list(range(d_min, d_max+1))
    for d in ds:
        dual = sp.dual_relaxation(d, sage_mults=True)
        dual.solve(verbose=verb)
        vals.append(dual.value)
        times.append(dual.timings['MOSEK']['solve_via_data'])
    print('\nBounds with lag = %s, A_strat = %s' % (lag, A_strat))
    for i, d in enumerate(ds):
        print('\tLevel %s: %s, \t %s' % (d, vals[i], times[i]))
    alldat = np.column_stack([ds, vals, times])
    return alldat, sp


def get_configs():
    f, gts, eqs, X = rm1978_p23(all_lag=True)
    gts = [mc.clear_den(g) for g in gts]
    params = [
        ('full', 'custom', 2, 3),
        ('no box', 'custom', 2, 4),
        ('minimal', 'custom', 2, 4),
        ('full', 'default', 1, 2),
        ('no box', 'default', 1, 3),
        ('minimal', 'default', 1, 3)
    ]
    return f, gts, eqs, X, params


def run_all_configs():
    f, gts, eqs, X, params = get_configs()
    filename = 'rm1978p23/sageres.pkl'
    alldat_dict = dict()
    for (lag, A_strat, d_min, d_max) in params:
        dat, sp = run_config(f, gts, eqs, X, lag, A_strat, d_min, d_max, False)
        alldat_dict[(lag, A_strat)] = (dat, sp)
    file = open(filename, 'wb')
    pickle.dump(alldat_dict, file)
    file.close()
    pass


def read_results():
    f, gts, eqs, X, params = get_configs()
    filename = 'rm1978p23/sageres.pkl'
    file = open(filename, 'rb')
    alldat_dict = pickle.load(file)
    file.close()
    return f, gts, eqs, X, params, alldat_dict


def format_alldat_dict(alldat_dict):
    num_trials = sum([v[0].shape[0] for v in alldat_dict.values()])
    alldat_arr = np.empty(shape=(num_trials, 5), dtype=object)
    i = 0
    for k, v in alldat_dict.items():
        datarr = v[0]
        for row in datarr:
            alldat_arr[i, 0] = k[0]
            alldat_arr[i, 1] = k[1]
            alldat_arr[i, 2] = int(row[0])
            alldat_arr[i, 3] = row[1]
            alldat_arr[i, 4] = row[2]
            i += 1
    alldat_arr[alldat_arr[:, 1] == 'custom', 1] = 'natural'
    alldat_arr[alldat_arr[:, 1] == 'default', 1] = 'naive'
    return alldat_arr


def sage_bounds_tables(ad, formatter="{:.3f}"):
    # formatted_float = "{:.2f}".format(a_float)
    # This function is hard-coded to assume that "ad" has sequences of rows
    # where a column reads 'full' and then 'no box' and then 'minimal'.
    # This function assumes that configurations marked 'full' and ONLY
    # those configurations have two values for degree (all others have
    # three values).
    tables = []
    # Want columns to give the bounds, rows indexed by degree.

    def get_row(deg, ring, idx):
        assert idx in {3, 4}
        adl = ad[(ad[:, 1] == ring) & (ad[:, 2] == deg), :]
        vals = adl[:, idx]
        valstrs = [formatter.format(v) for v in vals]
        if len(valstrs) == 2:
            # We're looking at a 'full' configuration and we skipped a run.
            valstrs = ['-'] + valstrs  # the first entry of the row of table corresponds to full configs.
        assert len(valstrs) == 3
        row = '%s & %s & %s & %s \\\\' % (deg, valstrs[0], valstrs[1], valstrs[2])
        return row

    cl1 = '$G_{\\text{all}}$'
    cl2 = '$G_{\\text{all}}\\setminus G_{\\text{box}}$'
    cl3 = '$G_{\\text{nonconvex}}$'

    tab = """
    \\begin{table}[ht!]
        \\begin{center}
        \\begin{tabular}{cccc}
        \\hline
        $G$ & %s & %s & %s \\\\
        $|\\cA|$ & %s & %s & %s \\\\ \\hline
        \\end{tabular}
        \\end{center}
        \\caption{Sizes of $\\cA$ used in naive rings for Problem \\ref{prob:ex1}.}
        \\label{tab:ex1:ring_sizes}
    \\end{table}
        """
    sizes = ring_support_sizes()
    t0 = tab % (cl1, cl2, cl3, sizes[2], sizes[1], sizes[0])
    tables.append(t0)

    t1 = """
    \\begin{table}[ht!]
        \\centering
        \\begin{tabular}{c|ccc} %% use \\textcolor{white}{1} to get proper spacing.
        $d$\\textbackslash $G$ & %s & %s & %s \\\\ \\hline
        %s
        %s
        %s \\hline
        \\end{tabular}
        \\caption{SAGE bounds for Problem \\ref{prob:ex1} using the natural signomial ring}
        \\label{tab:ex1:natural_bounds}
    \\end{table}
    """ % (cl1, cl2, cl3, get_row(2, 'natural', 3), get_row(3, 'natural', 3), get_row(4, 'natural', 3))
    tables.append(t1)
    t2 = """
    \\begin{table}[ht!]
        \\centering
        \\begin{tabular}{c|ccc} %% use \\textcolor{white}{1} to get proper spacing.
        $d$\\textbackslash $G$ & %s & %s & %s \\\\ \\hline
        %s
        %s
        %s \\hline
        \\end{tabular}
        \\caption{Solver runtimes for SAGE relaxations to Problem \\ref{prob:ex1}, using the natural signomial ring}
        \\label{tab:ex1:natural_solvetime}
    \\end{table}
    """ % (cl1, cl2, cl3, get_row(2, 'natural', 4), get_row(3, 'natural', 4), get_row(4, 'natural', 4))
    tables.append(t2)

    def get_wide_row(deg, ring, pairup):
        # assert idx in {3, 4}
        adl = ad[(ad[:, 1] == ring) & (ad[:, 2] == deg), :]
        bvals = adl[:, 3]
        tvals = adl[:, 4]
        bndstrs = [formatter.format(v) for v in bvals]
        timstrs = [formatter.format(v) for v in tvals]
        if len(bndstrs) == 2:
            # We're looking at a 'full' configuration and we skipped a run.
            bndstrs = ['-'] + bndstrs  # the first entry of the row of table corresponds to full configs.
            timstrs = ['-'] + timstrs
        assert len(bndstrs) == 3
        if pairup:
            row = '%s & %s & %s & %s & %s & %s & %s \\\\' % (deg,
                                                             bndstrs[0], timstrs[0],
                                                             bndstrs[1], timstrs[1],
                                                             bndstrs[2], timstrs[2])
        else:
            row = '%s & %s & %s & %s & %s & %s & %s \\\\' % (deg,
                                                             bndstrs[0], bndstrs[1], bndstrs[2],
                                                             timstrs[0], timstrs[1], timstrs[2])
        return row
    t3 = """
    \\begin{table}[ht!]
        \\centering
        \\begin{tabular}{c|cc|cc|cc} %% use \\textcolor{white}{1} to get proper spacing.
        $d$\\textbackslash $G$ & %s & %s & %s & %s & %s & %s \\\\ \\hline
        %s
        %s
        %s \\hline
        \\end{tabular}
        \\caption{Natural-ring SAGE bounds and solver runtimes (in seconds) for Problem \\ref{prob:ex1}.}
        \\label{tab:ex1:natural}
    \\end{table}
    """ % (cl1, 'time', cl2, 'time', cl3, 'time',
           get_wide_row(2, 'natural', True), get_wide_row(3, 'natural', True), get_wide_row(4, 'natural', True))
    tables.append(t3)
    t4 = """
    \\begin{table}[ht!]
        \\centering
        \\begin{tabular}{c|ccc|ccc|} %% use \\textcolor{white}{1} to get proper spacing.
        & \\multicolumn{3}{c}{$\\cA$-degree $d$ SAGE bounds} & \\multicolumn{3}{|c|}{solver runtimes (s)} \\\\
        $d$\\textbackslash $G$ & %s & %s & %s & %s & %s & %s \\\\ \\hline
        %s
        %s
        %s \\hline
        \\end{tabular}
        \\caption{Natural-ring SAGE bounds and solver runtimes (in seconds) for Problem \\ref{prob:ex1}.}
        \\label{tab:ex1:natural}
    \\end{table}
    """ % (cl1, cl2, cl3, 'time 1', 'time 2', 'time 3',
           get_wide_row(2, 'natural', False), get_wide_row(3, 'natural', False), get_wide_row(4, 'natural', False))
    tables.append(t4)
    t5 = """
    \\begin{table}[ht!]
        \\centering
        \\begin{tabular}{c|ccc|ccc|} %% use \\textcolor{white}{1} to get proper spacing.
        & \\multicolumn{3}{c}{$\\cA$-degree $d$ SAGE bounds} & \\multicolumn{3}{|c|}{solver runtimes (s)} \\\\
        $d$\\textbackslash $G$ & %s & %s & %s & %s & %s & %s \\\\ \\hline
        %s
        %s
        %s \\hline
        \\end{tabular}
        \\caption{Naive-ring SAGE bounds and solver runtimes (in seconds) for Problem \\ref{prob:ex1}.}
        \\label{tab:ex1:naive}
    \\end{table}
    """ % (cl1, cl2, cl3, cl1, cl2, cl3,
           get_wide_row(1, 'naive', False), get_wide_row(2, 'naive', False), get_wide_row(3, 'naive', False))
    tables.append(t5)
    return tables


def get_best_run_data(cond_sage=True):
    f, gts, eqs, X = rm1978_p23(all_lag=True)
    gts = [mc.clear_den(g) for g in gts]
    A = np.row_stack([np.eye(f.n), np.zeros(f.n)])
    if not cond_sage:
        X = None
    sp = lh.SignomialProgram(f, gts, eqs, X, A=A)
    dual = sp.dual_relaxation(3, sage_mults=True)
    return sp, dual


if __name__ == '__main__':
    f, gts, eqs, X, params, alldat_dict = read_results()
    ad = format_alldat_dict(alldat_dict)

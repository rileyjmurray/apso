# def get_model():
#     t = so.standard_sig_monomials(9)
#     k = {1: t[0],
#          2: t[1],
#          3: t[2],
#          4: t[3],
#          5: t[4],
#          7: t[5],
#          8: t[6]}
#     x = {2: t[7],
#          3: t[8]}
#     K = {
#         (0, 0): (1+k[2])*(1+k[4])*(1+k[5])*(1+k[8]),
#         (1, 0): (1+k[8])*(k[1] + k[3] + k[1]*k[4] + k[1]*k[5] + k[2]*k[3] + k[1]*k[4]*k[5]),
#         (0, 1): (1+k[2])*(k[3] + 4*k[7] + k[3]*k[5] + k[3]*k[8] + 4*k[4]*k[7] + k[3]*k[5]*k[8]),
#         (2, 0): k[1]*k[3]*(1+k[8]),
#         (1, 1): k[1]*(k[3] + 4*k[7] + k[3]*k[8] + 4*k[4]*k[7] - k[3]*k[5] - k[3]*k[5]*k[8]),
#         (0, 2): 4*k[3]*k[7]*(1+k[2]),
#         (1, 2): 4*k[1]*k[3]*k[7]
#     }
#     fk = K[(0, 0)] + K[(1, 0)]*x[2] + K[(0, 1)]*x[3] + K[(2, 0)]*x[2]**2 + K[(1, 1)]*x[2]*x[3]
#     fk += K[(0, 2)]*x[3]**2 + K[(1, 2)]*x[2]*x[3]**2
#     return fk, k, x
##
# clear all;
# clc;
# mset('yalmip',true)

using TSSOS
using DynamicPolynomials
@polyvar t[1:9]

## k = containers.Map({1,2,3,4,5,7,8}, {t[1],t[2],t[3],t[4],t[5],t[6],t[7]});

k = Dict(1 => t[1], 2 => t[2], 3 => t[3], 4 => t[4], 5 => t[5], 7 => t[6], 8 => t[7]);

## x = containers.Map({2,3}, {t[8], t[9]});

x = Dict(2 => t[8], 3 => t[9]);

K00 = (1+k[2])*(1+k[4])*(1+k[5])*(1+k[8]);
K10 = (1+k[8])*(k[1] + k[3] + k[1]*k[4] + k[1]*k[5] + k[2]*k[3] + k[1]*k[4]*k[5]);
K01 = (1+k[2])*(k[3] + 4*k[7] + k[3]*k[5] + k[3]*k[8] + 4*k[4]*k[7] + k[3]*k[5]*k[8]);
K20 = k[1]*k[3]*(1+k[8]);
K11 = k[1]*(k[3] + 4*k[7] + k[3]*k[8] + 4*k[4]*k[7] - k[3]*k[5] - k[3]*k[5]*k[8]);
K02 = 4*k[3]*k[7]*(1+k[2]);
K12 = 4*k[1]*k[3]*k[7];

fk = K00 + K10*x[2] + K01*x[3] + K20*x[2]^2 + K11*x[2]*x[3] + K02*x[3]^2 + K12*x[2]*x[3]^2;

k5 = 7.061;
w = 2.41;

# The natural specification of the problem.
#
#     (*) Bound constraints are linear inequalities.
#
pop_linear = [fk, #=
    =# (k[1] - w^-1), (w - k[1]), #=
    =# (k[2] - w^-1), (w - k[2]), #=
    =# (k[3] - w^-1), (w - k[3]), #=
    =# (k[4] - w^-1), (w - k[4]), #=
    =# (k[7] - w^-1), (w - k[7]), #=
    =# (k[8] - w^-1), (w - k[8]), #=
    =# x[2], #=
    =# x[3], #=
    =# k[5] - k5];

# The prefered specification of the problem when
# using the Lasserre hierarchy.
#
#     (*) Box constraints are converted to quadratic
#     inequalities.
#
#     (*) One-sided bound constraints are left as
#     linear inequalities.
#
pop_quad = [fk, #=
    =# (k[1] - w^-1) * (w - k[1]), #=
    =# (k[2] - w^-1) * (w - k[2]), #=
    =# (k[3] - w^-1) * (w - k[3]), #=
    =# (k[4] - w^-1) * (w - k[4]), #=
    =# (k[7] - w^-1) * (w - k[7]), #=
    =# (k[8] - w^-1) * (w - k[8]), #=
    =# x[2], #=
    =# x[3], #=
    =# (k[5] - k5)^2];


# An alternative specification of the problem
# that includes a constraint x[2]*x[3] >= 0.
#
#     (*) Using such a constraint (or other products
#     of constraints) means we're no longer working 
#     with the Lasserre hierarchy in its "pure" form.
#
pop_quad_bilinear = [fk, #=
    =# (k[1] - w^-1) * (w - k[1]), #=
    =# (k[2] - w^-1) * (w - k[2]), #=
    =# (k[3] - w^-1) * (w - k[3]), #=
    =# (k[4] - w^-1) * (w - k[4]), #=
    =# (k[7] - w^-1) * (w - k[7]), #=
    =# (k[8] - w^-1) * (w - k[8]), #=
    =# x[2], #=
    =# x[3], #=
    =# x[2]*x[3], #=
    =# (k[5] - k5)^2];


d = 3;
opt, sol, data = tssos_first(pop_quad_bilinear, t, d, numeq=1, TS="block");
# ^ Can try other representations, like pop_quad or pop_linear


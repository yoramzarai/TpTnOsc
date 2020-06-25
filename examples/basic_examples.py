'''
Few basic examples of the tools in TpTnOsc.utils.

See also osc_exp_examples.ipynb notebook for examples of tools in TpTnOsc.osc_exp.

Execute: python basic_examples.py
'''
import numpy as np
import TpTnOsc.utils as ut

# computing the p'th order multiplicative compound matrix
A = np.array([[1,6,0,0], [2,13,4,20], [2,13,5,25], [0,0,3,16]])
p = 2
mc, lp = ut.compute_MC_matrix( A, p )
print(f"mc=\n{mc}\nlp=\n{lp}")

print(f"\nA is I-TN: {ut.is_ITN(A)}")

# SEB factorization 
Lmat, Dmat, Umat, Um, valsL, valsU = ut.EB_factorization_ITN(A)
print(f"\nvalsL={valsL}, valsU={valsU}")

# generare an oscillatory matrix
valsL = np.array([1,0,1,2,1,0])
valsU = np.array([1,3,2,3,0,0])
valsD = np.array([2,1,4,2])
B = ut.compute_matrix_from_EB_factorization( valsL, valsD, valsU )
print(f"\nB=\n{B}\nB is OSC={ut.is_OSC(B, tol=10*np.finfo(np.float).eps)}")
print(f"\nSEB factorization = {ut.show_EB_config(valsL, valsU, valsD, True)}")

# format matrix in latex form
print("\nB in Latex form:")
ut.show_mat_latex_format(B)

# sign variations
v = np.array([-1,2,-3,0])
print(f"\ns-(v)={ut.s_minus(v)}\ns+(v)={ut.s_plus(v)}")
print(f"\nsc-(v)={ut.sc_minus(v)}\nsc+(v)={ut.sc_plus(v)}")


# draw SEB factorization
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 6))
ut.draw_EB_factorization_ITN( valsL, valsD, valsU, ax)
plt.show()


# computing the exponent of an oscillatory matrix
import TpTnOsc.osc_exp as oscp

print("\n\nComputing families of vertex-disjoing paths and exponent (r()) of B:\n")
osc_cls = oscp.OSC_exp(B)
osc_cls.run()
print("lower-left and upper-right corner minors families of vertex-disjoint paths:")
osc_cls.display_results()

for k in osc_cls.G.keys():
    print(f'Triangle graph of {k}:')
    _, ax = plt.subplots(figsize=(9,6))
    osc_cls.G[k].draw_me(ax, nd_offset=[0.3, 0.4])
    ax.margins(.1, .1)
    plt.show()
    plt.close()

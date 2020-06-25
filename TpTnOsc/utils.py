'''
General tools related to totally positive (TP), totally nonnegative (TN), invertible-TN (I-TN) 
and oscillatory (OSC) matrices. 

In addition, functions to compute the number of sign variations in a vector (both
cyclic and non-cyclic) are provided.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
import networkx as nx
import functools as fnt
from itertools import combinations, product

'''Implementation here of the EB factorization on an invertible TN matrix
    is based on chapter 2 of the book "Totally Nonnegative Matrices", Fallat & Johnson.'''

def matrix_minor(A, r_indxs, c_indxs):
    '''This function returns the minor of a matrix A with rows indexed by r_indxs and 
    columns by c_indxs. r_indxs and c_indxs are lists (or 1D numpy arrays), and these indexes 
    start from 0 (which is the first row/column index).'''
    return np.linalg.det(A[np.ix_(r_indxs, c_indxs)]) if len(r_indxs)==len(c_indxs) else None

def compute_MC_matrix( A, p ):
    '''This function computes the p'th order multiplicative
    compound matrix of the given matrix A. It returns the MC
    matrix and the lexicography order (with 0 as the first index)'''
    x = np.arange(np.minimum(*A.shape), dtype=int)  # 0, .., n-1, where n:=min(matrix dimensions)
    lp = np.array(list(combinations(x, p))) # lexicography order of the p inxedes in x 
    lp_len = len(lp)
    Q = np.array([matrix_minor(A, lp[r], lp[c]) for r in range(lp_len) for c in range(lp_len)]).reshape(lp_len, lp_len)
    return Q, lp

def E(n, i, j):
    '''Returns the E_{ij} matrix'''
    mat = np.zeros((n,n))
    mat[i-1,j-1] = 1
    return mat

def L(n, i, a):
    '''Returns the L_i(a) matrix'''
    return np.identity(n) + a*E(n, i, i-1)

def U(n, i, a):
    '''Returns the U_i(a) matrix'''
    return np.identity(n) + a*E(n, i-1, i)

def compute_L_factorization( A, abs_thres=0 ):
    '''This function computes the left-hand side of the SEB factorization
    of a square matrix.
    
    Given a matrix A, the function uses the Neville elimination algorithm
    to compute L and U, such that A = LU, where L:=[L_n*L_{n-1}*..*L_2]*..*[L_n],
    and U is an upper-triangular matrix.
    
    The outputs are:
     1. A list of the L_i matrices (in the factorization order)
     2. The matrix U.
     3. The parameter values of the L_i matrices.
    '''
    n = A.shape[0]
    k = comb(n, 2, exact=True)
    Lmat = []
    vals = []
    Um = A
    # Neville algorithm
    for j in range(n-1):
        for i in range(n-1,j,-1):
            val = Um[i,j] / Um[i-1,j] if Um[i-1,j] != 0 else 0
            if np.abs(val) < abs_thres: # < 10*np.finfo(np.float).eps:
                val = 0
            vals.append(val)
            Lmat.append(L(n,i+1, val))
            Um = np.matmul( L(n,i+1, -val), Um )
    return Lmat, Um, np.asarray(vals)

def EB_factorization_ITN( A, abs_thres=0 ):
    '''This function computes the EB factorization of 
    an inversed TN matrix. See Theorem 2.2.2 for more details.
    
    Given an inversed TN matrix A, the following holds:
    
       A = Lmat{1}*Lmat{2}*...*Lmat{end}*Dmat*Umat{end}*Umat{end-1}*...*Umat{1}.
       
     For example, for n=4:
       A = L_4(l_1)*L_3(l_2)*L_2(l_3)*L_4(l_4)*L_3(l_5)*L_4(l_6)*D*
           U_4(u_6)*U_3(u_5)*U_4(u_4)*U_2(u_3)*U_3(u_2)*U_4(l_1),
       
       
    Returned parameters:
    Lmat - a list of the L matrices in the order as in the multiplication. 
           For example, for n=4: [L_4(valsL(1)),L_3(valsL(2)),L_2(valsL(3)),L_4(valsL(4)),L_3(valsL(5)),L_4(valsL(6))].
    Dmat - the diagonal matrix.
    Umat - a list of the U matrices in the REVERSED order of the multiplication.
           For example, for n=4: [U_4(valsU(1)),U_3(valsU(2)),U_2(valsU(3)),U_4(valsU(4)),U_3(valsU(5)),U_4(valsU(6))].
    
    valsL - the l_i values corresponding to the order of the L matrices in the multiplication: L_n*..*L_2*L_n*...L_3*...L_n
    valsU - the u_i values corresponding to the REVERSED order of the U matrices in the multiplication: U_n*U_{n-1}*U_n*...*U_2*U_3*...*U_n.
   
    For example, for a 4x4 matrix A  we have
    
     A = Lmat{1}(valsL(1))*Lmat{2}(valsL(2))*...*Lmat{6}(valsL(6))*Dmat*
         Umat{6}(valsU(6))*Umat{5}(valsU(5))*...*Umat{1}(valsU(1)).       
    '''
    if A.shape[0] != A.shape[1]:
        print('Error: input matrix must be square for EB factorization of an ITN matrix !!')
        return
    Lmat, Um, valsL = compute_L_factorization( A, abs_thres )
    Umat_tmp, Dmat, valsU = compute_L_factorization( Um.transpose(), abs_thres )
    Umat = [x.transpose() for x in Umat_tmp]
    return Lmat, Dmat, Umat, Um, valsL, valsU 

def compute_L_indexes( n ):
    '''This function computes the L matrix indexes. For example,
    for n=4, the indexes are [4 3 2 4 3 4]'''
    xbase = np.array(range(n,1,-1))
    x = xbase
    for i in range(1,n-1):
        x = np.concatenate((x,xbase[:-i]))
    return x

def display_EB_factorization( Lmat, Dmat, Umat, valsL, valsU ):
    '''This function displays the factorization matrices in
    the order of the factorization multiplication (left to right).
    For the exact order of each input parameter, see the function EB_factorization_ITN()
    '''
    n = Lmat[0].shape[0]
    idxs = compute_L_indexes( n )
    k = idxs.shape[0]
    print("Factorization matrices in the order as in the factorization form (left-most to right-most matrix):")
    # L matrices
    for i in range(k):
        print("L{0}({1:4.2f})=\n{2}".format(idxs[i], valsL[i], Lmat[i]))
    # D matrix
    print("D=\n{}".format(Dmat))
    # U matrices
    idxs = np.flip( idxs )
    valsu = np.flip( valsU )
    for i in range(k):
        print("U{0}({1:4.2f})=\n{2}".format(idxs[i], valsu[i], Umat[(k-1)-i]))
    
def EB_factorization_k2n(k):
    '''This function returns the n value given k. k is the number of L and U parameters
    in the EB factorization on a square I-TN matrix of size n.
    n = (1+sqrt(1+8*k))/2.'''
    return int((1 + np.sqrt(1+8*k))/2)

def EB_factorization_n2k(n):
    '''This function returns the k value given n. k is the number of L and U parameters
    in the EB factorization on a square I-TN matrix of size n.
    k = ((n-1)*n)/2'''
    return int(((n-1)*n)/2)

def lexichog_order(n, p):
    '''This function returns the p'th order lexicography indxes array based on 
    the array 0, ..., n-1.
    
    For example, for n=4 and p=2, the function returns:
    np.array[[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]'''
    return np.array(list(combinations(np.arange(n, dtype=int), p))) # lexicography order of the p inxedes in 0, ..., n-1

def lexicog2linear(n, r, c):
    '''This function converts a lexicography matrix index to a linear index. 
     The function assumes that all indexes starts from 0.
     
    Inputs:
    r = [r_1, r_2,..., r_p]
    c = [c_1, c_2,..., c_p]
    where r_i, c_i get values between 0 to n-1.
    
    The function returns the tuple (i, j) correponding to row i and column j of
    r and c, respectively (where 0 in the first row/column).
    
    For example, for n=4, r=[0,3], c=[1,3] we get i=3, j=4. '''
    if len(r) != len(c):
        print('Error: r and c length missmatch !!')
        return
    lp = lexichog_order(n, len(r))  # np array of lexicography order
    kvec = np.arange(len(lp))
    return kvec[(lp==r).all(axis=1)][0], kvec[(lp==c).all(axis=1)][0]

def linear2lexicog(n, p, i, j):
    '''This function converts a linear index to a lexicography index.
    
    For example, for n=4, p=3, i=2, and j=0 we get r=[0,2,3], c=[0,1,2]  
    '''
    lp = lexichog_order(n, p)  # np array of lexicography order
    if (i>=len(lp)) or (j>=len(lp)):
        print('Error: i and/or j larger than {} !!'.format(len(lp-1)))
        return
    return lp[i], lp[j]

    
def draw_EB_factorization_ITN( valsL, d, valsU, ax, 
                               compress_f = True, font_size=24, font_color='r', perc_round=4, 
                               base_weight=1, tol=10*np.finfo(np.float).eps, noffset=0.2 ):
    '''This function draws the graph corresponding to the given EB factorization (in the 
    form of the L matrix parameters, the digonal of the diagonal natrix and the U 
    matrix parameters). The function supports compressing the graph in the sense of removing
    L and/or U matrices with parameters equal to zero.
    
    Inputs:
    valsL, valsU - see the output parameters of the function EB_factorization_ITN()
    d - the diagonal of the diagonal matrix D, i.e. [d_{11},d_{22},...,d_{nn}] 
    '''
    n = EB_factorization_k2n(valsL.shape[0]) #int((1 + np.sqrt(1+8*k))/2)
    idxs = compute_L_indexes( n )
    
    if compress_f: # remove L/U matrices with zero parameters
        locl = valsL!=0
        locu = valsU!=0
    else: 
        locl = np.ones(valsL.size, dtype=bool)
        locu = np.ones(valsU.size, dtype=bool)
            
    vL = valsL[locl]
    lidxs = idxs[locl]  # indexes corresponding to vL
    nvL = vL.size
    vU = valsU[locu]
    uidxs = idxs[locu]  # indxes corresponding to vU
    nvU = vU.size
    num_h_nodes = nvL+nvU+2 # total number of horizontal nodes
    
    #G = nx.Graph()  # for undirected graph
    G = nx.DiGraph() # for directed graph
    # all nodes in the graph (total of n rows and num_h_nodes columns)
    for j in range(num_h_nodes):
        for i in range(n):
            G.add_node(j*n+i,pos=(j,i))
    # edges corresponding to the L matrices   
    for j in range(nvL):
        if(np.abs(vL[j]) > tol): # L_k(m) adds an edge from node k to node k-1 of weight m
            G.add_edge(j*n+lidxs[j]-1,(j+1)*n+lidxs[j]-2, weight=vL[j])
        for i in range(n): # all horizontal edges of weight 1
            G.add_edge(i+j*n,(j+1)*n+i, weight=base_weight)         
    # horizontal edges corresponding to the D matrix
    for i in range(n):
        G.add_edge(i+nvL*n,i+(nvL+1)*n, weight=d[i])

    # edges corresponding to the U matrices   
    vu = np.flip(vU)
    uidxs = np.flip(uidxs)
    for j in range(nvL+1,num_h_nodes-1):
        m = j-(nvL+1) # the corresponding index in uidxs and vu
        if(np.abs(vu[m]) > tol): # U_k(m) adds an edge from k-1 to k of weight m
            G.add_edge(j*n+uidxs[m]-2,(j+1)*n+uidxs[m]-1, weight=vu[m])
        for i in range(n): # all horizontal edges of weight 1
            G.add_edge(j*n+i,(j+1)*n+i, weight=base_weight)

    nn = np.array(range(1,n+1))
    lnames = {k:v for (k,v) in enumerate(nn)}
    rnames = {k:v for (k,v) in zip( range((num_h_nodes-1)*n,(num_h_nodes*n)), nn)}
    nnames = {**lnames, **rnames} # node names
    pos = nx.get_node_attributes(G,'pos')
    nx.draw(G, pos, ax=ax)
    edge_labels={(u,v,):round(d['weight'],perc_round) for u,v,d in G.edges(data=True)}
    
    #nx.draw_networkx_edges(G, pos)  # ADDED
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax);
    
    
    # node labels (names) - we shift the position of the names of the source and sink nodes to the left and
    # right, respectively.
    pos_n = pos.copy()
    for k in range(n):
        for (o,v) in zip([0,n*(num_h_nodes-1)], [-noffset, noffset]):
            pos_n[k+o] = (pos_n[k+o][0]+v, pos_n[k+o][1])
    nx.draw_networkx_labels(G, pos_n, ax=ax, labels=nnames, font_size=font_size, font_color=font_color);
    

def compute_matrix_from_EB_factorization( valsL, valsD, valsU ):
    '''This function multiplies all factorization matrices corresponding to the
    factorization parameters given to the function, to obtain the original matrix.
    Basicall, the function computes:
     A = (L_n(valsL_1)*..*L_2(valsL_{n-2}))*(L_n(valsL_{n-1})*..)..*(L_n(valsL_k))*diag(valsD)*
         (U_n(valsU_k))*(U_{n-1}(valsU_{k-1})*U_n(valsU_{k-2}))*.....*U_n(valsU_1).
         
    For example, for n=4, the function computes:
    A = L_4(valsL_1)*L_3(valsL_2)*L_2(valsL_3)*L_4(valsL_4)*L_3(valsL_5)*L_4(valsL_6)*diag(valsD)*
        U_4(valsU_6)*U_3(valsU_5)*U_4(valsU_4)*U_2(valsU_3)*U_3(valsU_2)*U_4(valsU_1).     
    '''
    k = valsL.shape[0]
    n = EB_factorization_k2n(k) #int((1 + np.sqrt(1+8*k))/2)
    idxs = compute_L_indexes( n )
    
    # product of all L matrices, multiplied by D, multiplied by the product of all U matrices
    return fnt.reduce(np.matmul, [L(n, idxs[i], valsL[i]) for i in range(k)]) @ \
           np.diag(valsD) @ \
           fnt.reduce(np.matmul, [U(n, idxs[i], valsU[i]) for i in reversed(range(k))])

def show_EB_config( valsL, valsU, valsD=0, mode=False ):
    '''This function returns the EB factorization configuration, in a form of a string,
    given the L and U matrices parameters. If mode==False (default), the L and U
    parameters are not displayed, otherwise they are displayed together with the diagonal
    entries of the matrix D (valsD). 
    For the exact order of valsL and valsU parameters, see the function EB_factorization_ITN().
    
    For example, 
    show_EB_config( np.array([1,0,5,0,9,0]), np.array([0,.1,0.3,0.7,0,0]), np.array([1,2,3,4]), True ) yields:
    'L4(1)*L2(5)*L3(9)*D([1 2 3 4])*U4(0.7)*U2(0.3)*U3(0.1)', 
    and show_EB_config( np.array([1,0,5,0,9,0]), np.array([0,.1,0.3,0.7,0,0])) yields:
    'L4*L2*L3*U4*U2*U3'.
    '''
    idxs = compute_L_indexes( EB_factorization_k2n(valsL.shape[0]) )
    sr = ''
    loc = valsL!=0
    vl = valsL[loc]
    ids = idxs[loc]
    for i in range(len(vl)):  # the L matrices
        sr += 'L'+str(ids[i])
        if mode: sr += '('+str(vl[i])+')'
        sr += '*'    
    if mode:  # The D matrix
        sr += 'D('+str(valsD)+')*'
    loc = valsU!=0
    vl = np.flip(valsU[loc])
    ids = np.flip(idxs[loc])
    for i in range(len(vl)):  # the U matrices
        sr += 'U'+str(ids[i])
        if mode: sr += '('+str(vl[i])+')'
        sr += '*' 
    return sr[:-1]

def is_TP( A, tol=10*np.finfo(np.float).eps ):
    '''This function returns True [False] if A is [is not]
    a TP matrix. A matrix is TP is all MC are > tol'''
    return all([(compute_MC_matrix(A, p)[0]>tol).all() for p in range(1,A.shape[0]+1)])

def is_TN( A ):
    '''This function returns True [False] if A is [is not]
    a TN matrix.'''
    return all([(compute_MC_matrix(A, p)[0]>=0).all() for p in range(1,A.shape[0]+1)])

def is_invertible( A, tol=10*np.finfo(np.float).eps ):
    '''This function returns True [False] if A is [is not]
    an invertible matrix. A matrix is invertible if det(A)>tol'''
    return (A.shape[0]==A.shape[1]) and (np.abs(np.linalg.det(A))>tol)

def is_ITN( A, tol=10*np.finfo(np.float).eps ):
    '''This function returns True [False] if A is [is not]
    an inversible TN matrix.'''
    return is_TN(A) and is_invertible(A, tol)
           
def is_OSC( A, tol ):
    '''This function returns True [False] if A is [is not]
    an oscillatory matrix.'''
    return is_ITN(A, tol) and is_TP(np.linalg.matrix_power(A, A.shape[0]-1), tol)

def is_factorization_osc(lvals, uvals, dvals, lindxs = None):
    '''This function checks if the given factorization (given by the l, u, and d parameters)
    results in an oscillatory matrix.'''
    n = EB_factorization_k2n(lvals.shape[0])
    if lindxs is None:
        lindxs = compute_L_indexes(n)
    return (dvals>0).all() and all([ (lvals[j]>0).any() and (uvals[j]>0).any() for i in lindxs for j in np.where(lindxs==i)])

def is_factorization_TP(lvals, uvals, dvals, lindxs = None):
    '''This function checks if the given factorization (given by the l, u, and d parameters)
    results in a TP matrix.'''
    return (dvals>0).all() and (lvals>0).all() and (uvals>0).all()

def is_osc_from_factorization(A: list) -> bool:
    '''
    This function checks if the input matrix is oscillatory by examining the 
    matrix SEB factorization.
    '''
    _, Dmat, _, _, lvals, uvals  = EB_factorization_ITN( A )
    return is_factorization_osc(lvals, uvals, np.diagonal(Dmat))

def show_mat_latex_format(A, fmt='4f'):
    '''This function prints a matrix in a latex format
    to the screen.'''
    print('\\begin{bmatrix}')
    for j, row in enumerate(A,1):
        for x in row[:-1]:
            print(f'{x:.{fmt}}', end=' & ')
        print(f"{row[-1]:.{fmt}}", end='')
        if j < A.shape[0]:  print(" \\\\")
    print('\n\\end{bmatrix}')
    
def osc_exp(A, tol=0):
    '''Returns the exponent of the oscillatory matrix A. 
    It is assumed that A is oscillatory (i.e. no checking is done).'''
    for r in range(1,A.shape[0]):
        if(is_TP(np.linalg.matrix_power(A,r), tol)):
            break
    return r

''' Number of sign variations functions'''
def s_minus(v):
    '''This function computes s^{-}(v), where v\in\R^{n} is a numpy array.'''
    return np.sum(np.abs(np.diff(np.sign(v[v!=0])))/2, dtype=np.int16)

def sc_minus(v):
    '''This function computes s_c^{-}(v) (cyclic number of sign variations), 
    where v\in\R^{n} is a numpy array.'''
    sm = s_minus(v)
    return sm+np.mod(sm,2)
    
def s_plus(v):
    '''This function computes s^{+}(v), where v\in\R^{n} is a numpy array.'''
    if (loc := np.nonzero(v==0)[0]).size > 0:
        allcomb = product([1,-1], repeat=len(loc))
        m = 0
        vv = np.copy(v)
        for i in allcomb:
            np.put(vv,loc,i)  # same as vv[loc]=i
            m = max(m, s_minus(vv))
        return m
    else:
        return(s_minus(v))

def sc_plus(v):
    '''This function computes s_c^{+}(v), where v\in\R^{n} is a numpy array.'''
    sp = s_plus(v)
    return sp+np.mod(sp,2)


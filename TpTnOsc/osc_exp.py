'''
Evaluating the exponent of oscillatory matrices.
'''

from collections import defaultdict
import numpy as np
import pprint
import itertools as itr
import copy 
import re
import networkx as nx
import TpTnOsc.utils as tpf

'''
Functions for lower-left corner minor paths.
=============================================
'''

def draw_triag_graph(tbl: list, ax, noffset: list=[0.1, 0.3], font_size: int=24, font_color: str='k', \
            edge_weight=2, arrow_size=2, node_size: int=200) -> None:
    '''
    This function draws the triangle graph in the matplotlib Axis object ax.
    
    For the input tbl, see the tf_tbl attribute of Trig_G() class.
    noffset = [x,y], where x is the label offset for the nodes on the left, and
    y is the label offset for the nodes on the bottom.
    
    '''
    n = tbl.shape[0]+1
    G = nx.DiGraph()  # directed graph
    
    # add the network nodes
    cnt = itr.count()
    for i in range(n):
        for j in range(n-i):
            G.add_node(next(cnt), pos=(i,j))
            
    # add all diagonal edges
    cnt, delta = 1, n-1
    for i in range(n-1):
        for _ in range(1,n-i):
            G.add_edge(cnt, cnt+delta, weight=1, color='b')
            cnt += 1
        cnt, delta = cnt+1, delta-1
    
    # add down edges based on tbl input
    cnt = 1
    for i in range(n-1):
        for j in range(1,n-i):
            if tbl[j-1,i]:
                G.add_edge(cnt, cnt-1, weight=1, color='r')
            cnt += 1
        cnt += 1

    # left side node names
    lnames = {k:v for (k,v) in zip(np.array(range(1,n)), np.array(range(2,n+1)))}
    # bottom side node names
    cs = np.append([0],np.cumsum(np.array(range(n, 1, -1)))) # [0,n,n+(n-1),...,n+(n-1)+...+2]
    bnames = {k:v for (k,v) in zip(cs, np.array(range(1,n+1)))}
    # node names
    nnames = {**lnames, **bnames} # merging dictionaries
    
    # offset node labels
    pos = nx.get_node_attributes(G,'pos')
    pos_n = pos.copy()
    # left names offset
    for k in range(1, n):
        pos_n[k] = (pos_n[k][0]-noffset[0], pos_n[k][1])
    # bottom names offset
    for k in cs:
        pos_n[k] = (pos_n[k][0], pos_n[k][1]-noffset[1])
    
    nx.draw(G, pos, ax=ax, node_size=node_size, edge_color=nx.get_edge_attributes(G,'color').values(),\
            edge_width=edge_weight, arrowsize=arrow_size)
    nx.draw_networkx(G, pos, ax=ax, node_size=node_size, \
                     edge_color=nx.get_edge_attributes(G,'color').values(), with_labels=False, width=edge_weight)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size)
    nx.draw_networkx_labels(G, pos_n, ax=ax, labels=nnames, font_size=font_size, font_color=font_color);

def valsL2L_info(valsL: list) -> list:
    '''This function generates L_info given the
    L_i parameter values corresponding to [L_n*...*L_2]*[L_n*...*L_3]*...*[Ln].
    
    L_info is a list of lists. Each list element in L_info contains all the i indexes
    for which L_i exists in the EB factorization. The first list element in info
    corresponds to [L_n,...,L_2], the second to [L_n,...,L_3], etc.
    
    For example, for n=4 and left-hand side factorization [L5_L_4L_2][L_5L_3][L_5L_4][],
    the corresponding L_info is [[5,4,2],[4,3],[5,4],[]].
    
    See the function EB_factorization_ITN() in TP_TN_OSC_funcs.py that 
    generates valsL given an I-TN matrix A = LDU.
    Note that valsL here can also be equal to valsU that is also generated by 
    EB_factorization_ITN() in order to generate the L_info of U^T.
    '''
    n = tpf.EB_factorization_k2n(len(valsL))
    all_L = list(tpf.compute_L_indexes(n))
    cms = np.r_[[0], np.cumsum([x for x in range(n-1,0,-1)])]  # [0 cumsum([n-1,...,1])]
    L_info = []
    for i in range(n-1):
        g = all_L[cms[i]:cms[i+1]]      # the corresponding section in the factorization
        v = valsL[cms[i]:cms[i+1]] > 0  # the corresponding L_i with parameter>0
        L_info.append([k for (k, f) in zip(g, v) if f])
    return L_info

def is_L_info_valid_ITN(info: list) -> bool:
    '''This function checks if info corresponds
    to a left-hand size of the SEB factorization of an I-TN matrix. 
    It simply checks that info contains at least one instant of the values
    2,...,n.'''
    flat = set([i for j in info for i in j])
    return all([k in flat for k in range(2,len(info)+2)])

def init_tri_tbl(L_info: list):
    '''This function initializes a triangle table based on the left-hand side
    of the SEB factorization.
    
    The returned numpy array contains the True/False triangle table located 
    upside-down in its upper-left side. All the other elements in the returned
    table (i.e. elements outside the upside-down triangle) are set to False.
    
    For example, for L_info = [[5,4,2],[4,3],[5,4],[]], the returned array is:
    
    [[ True  True  True False]
     [False  True  True False]
     [ True False False False]
     [ True False False False]]
     
     i.e. the upside-down True/False triangle table is:
     
      True  True  True  False
     False  True  True
      True False 
      True
      
      which correspondonds to the (not upside-down) triangle table:
      
      O
      O X
      X O O 
      O O O X
    '''
    n1 = len(L_info)
    tbl = np.zeros((n1,n1), dtype=bool)

    for x in range(n1):
        for y in range(n1+1,x+1,-1):
            '''we check if n,n-1,...,x+2 exists in L_info[x],
            where x = 0,1,...,n1-1'''
            tbl[y-2-x,x] = y in L_info[x] 
    return tbl

def view_tri_tbl(tri_tbl: list, pre: str ='', lb: dict={True: 'O', False: 'X'}, verbose: bool=True, dig_width: int=2, num_space: int=1) -> None:
    '''This function displays the triangle graph tri_tbl.'''
    s = ' '
    for show, y in enumerate(range(tri_tbl.shape[0]-1, -1, -1), start=1):
        print(pre, end='')
        if verbose:
            print(f"{y+2:{dig_width}}{s*num_space}", end='')
        for x in range(show):
            print(f"{lb[tri_tbl[y,x]]}", end=s*num_space)
        print()
        
    if verbose:
        print(f"{pre}{s*(num_space+dig_width)}", end='')
        for i in range(tri_tbl.shape[0]):
            print(f"{i+1}", end=s*num_space)
        print()
        
def view_tri_tbl_d(tri_tbl: list, pre: str ='', lb: dict={True: '\u028c', False: '\u2798'}, verbose: bool=True, dig_width: int=2, num_space: int=1) -> None:
    '''This function displays the triangle graph tri_tbl with the destination row.'''
    s = ' '
    for show, y in enumerate(range(tri_tbl.shape[0]-1, -1, -1), start=1):
        print(pre, end='')
        if verbose:
            print(f"{y+2:{dig_width}}{s*num_space}", end='')
        for x in range(show):
            print(f"{lb[tri_tbl[y,x]]}", end=s*num_space)
        print()
        
    if verbose:
        print(f"{1:{dig_width}}{s*num_space}", end='')
    for i in range(tri_tbl.shape[0]+1):
        print(f"\u25a1", end=s*num_space)
    print()
    if verbose:
        print(f"{pre}{s*(num_space+dig_width)}", end='')
        for i in range(tri_tbl.shape[0]+1):
            print(f"{i+1}", end=s*num_space)
    print()

def gen_LL_minor_indexes(n: int, x: int):
    '''This function generates a lower-left minor indexes. 
    The indexes in the minor are n-x+1,...,n|1,...,x. 
    This is returned as a numpy array np.array([[n-x+1,...,n], [1,...,x]])
    '''
    return np.array([[i for i in range(n-x+1,n+1)],[i for i in range(1,x+1)]])

def minor_indexes2str(mnr) -> str:
    '''This function converts the numpy 2D array of the minor indexes to a string
    "s|d", where s contains the source indexes and d the destination indexes.
    For example minor_indexes2str([[5,6],[1,2]]) results in '5,6|1,2'.
    '''
    return f"{','.join(map(str, mnr[0,:]))}|{','.join(map(str, mnr[1,:]))}"

def index_path(tf_tbl, d: int, oy: int=1, ox: int=0) -> list:
    '''This function computes the path of a source index to its destination
    index in the triangle graph. d here is the destination (1D) index.
    The start 2D index in the graph is (y,x)=(tf_tbl.shape[0],1) and we move 
    along the triangle graph either down (if we can, i.e.
    if there is no X) or diagonally (in which case we increase 
    the x axis index). We stop once we reach a destination (i.e. 
    if (y+1)+x=1+d).
    '''
    x, p = 1, []
    for y in range(tf_tbl.shape[0],0,-1):
        p.append([y,x])
        '''We move diagonally (corresponkding to horizontal
        path in the planar network) if:
        1. We encountered X (i.e. False), or
        2. We reached the destination 2D index. A destination 2D index
           is [1,d], and we reached a destination index if (y+1)+x = 1+d.
           Recall that here y values are: n-1,n-2,...,1, thus we need to add
           1 to y to get the real 2D index.'''
        if (not tf_tbl[y-1,x-1]) or (x+y==d):
            x += 1  
            
    # convert to indexes corresponding to the triangle graph.
    return path_offset(p+[[y-1,x]], oy, ox)      


def update_tf_tbl(po: list, tri_tbl) -> None:
    '''This function adds X (i.e. False) in the 2D indexes that are above the given path po. 
    An index above (y,x) is defined by (y+1,x). 
    Note that in po, y take values from 2,3,...,n, and x take values from 1,2,...,n-1.
    Thus, in the numpy triangle table tri_tbl (where y values are 0,1,...,) we need to 
    decrease y by 1 in order to mark the 2D index above the corresponding y indexes in po.
    '''
    for y,x in po:
        tri_tbl[y-1,x-1] = False  # the 2D index above the indexes in the path po
        #tri_tbl[y-2,x-1] = False # the indexes in the path po

def tf_tbl_trig_slice(tf_tbl, y: int):
    '''Given a value y (y \in\{2,3,...,n\}), the function returns the triangle sub-table
    whos top-left 2D index is (y,1).'''
    return tf_tbl[0:y-1,0:y-1]

def path_offset(p: list, oy: int, ox: int=0) -> list:
    '''This function offsets the y and x indexes in the path p.'''
    po = copy.deepcopy(p)
    for i in range(len(p)):
        po[i][0] += oy
        po[i][1] += ox

    return po

def path_2to1(path:list) ->list:
    '''This function converts a 2D representation of a path to 1D representation.
    For example path_2to1([[5,1],[4,2],[3,2],[2,2],[1,3]]) returns [5,5,4,3,3]'''
    return [x+y-1 for (x,y) in path]

def show_1dpath(path1d: list, flip=None, compress=None) -> str:
    '''This function returns a string in the form x1->x1->...x_n, 
    for path1d = [x1,x2,...,xn]. If flip is not None then
    the returned string is xn->...->x1.
    If compress is not None, the returned string is x1-->xn (or xn-->x1)'''
    compress = False if compress is None else True
    p = path1d.copy()[::-1] if flip else path1d
    s = ''
    if compress:
        s = f"{p[0]}-->"
    else:
        if len(p)==1:
            s = f"{p[0]}->" 
        for x in p[:-1]:
            s += f"{x}->"

    return s+f"{p[-1]}"

def horizontal_path(idx: int) -> list:
    '''This function returns a horizontal path in the triangle graph.
       For example, if idx=3, then the horizontal path is [[3,1], [2,2], [1,3]].
       '''
    return [[idx-i,i+1] for i in range(idx)]
    
def flip_str_by_sp(llm: str, sp: str) -> str:
    '''Given a string input the function flips the string around a separator (sp).
    For example flip_str_by_sp("3,4,5|1,2,3", "|") yields "1,2,3|3,4,5".
    '''
    z = re.match(f"(.+)\{sp}(.+)", llm)
    return f"{z.group(2)}{sp}{z.group(1)}"
    
def run_ll_minor(mnr, mnr_tbl, verbose: bool=False, MAX_copies: int=100) -> tuple:
    '''This function computes the paths of a lower-left minor given a triangle graph.
    Inputs: 
       mnr: a lower-left minor indexes, i.e. n-x+1,...,n|1,...,x, in the
            form of a numpy array [[n-x+1,...,n],[1,...,x]]
       mnr_tbl: the triangle table corresponding to the minor indexes.
       
    Outputs:
       num_copies - minimum number of matrix copies in order to obtain a single family of
                    vertex-disjoint paths.
       paths - a dictionary of paths, where keys are source->destination indexes.
    '''
    tf_tbl = np.copy(mnr_tbl)
    mnr_str = minor_indexes2str(mnr)
    done, num_copies = False, 1
    cur_mnr = mnr.copy()
    num_idxs = mnr.shape[1]  # number of indexes in the minor
    dest = np.zeros(num_idxs, dtype=int)  # container for destination of each index
    idxs_done = [False for i in range(num_idxs)]
    paths = defaultdict(list)
    vprint = print if verbose else lambda *a, **k: None
    
    vprint(f"\nMinor {mnr_str}:\n-----------------------")
    while not done:
        vprint(f'\n\tCopy {num_copies}, current graph:')
        view_tri_tbl(mnr_tbl, '\t') if verbose else None
        for k in range(num_idxs):
            cur_idxs = cur_mnr[:,k]
            vprint(f"\n\tPath for {cur_idxs[0]}->{cur_idxs[1]}...")
            if cur_idxs[0] > cur_idxs[1]: 
                # have yet to reach the destination vertex, i.e. source > destination
                sl_tbl = tf_tbl_trig_slice(mnr_tbl, cur_idxs[0])
                vprint(f"\tCorresponding graph:")
                view_tri_tbl(sl_tbl, '\t') if verbose else None
                po = index_path(sl_tbl, cur_idxs[1])
                if k < num_idxs - 1: # no need to update the table after the last index
                    update_tf_tbl(po, mnr_tbl)
            else:
                '''use horizontal path as we are already in the destination. For
                example, if cur_indxs[0]=cur_indxs[1]=3, then the horizontal path
                is [[3,1], [2,2], [1,3]] '''
                po = horizontal_path(cur_idxs[1])
            vprint(f"\t{po=}")
            if (d:=po[-1][1]) == cur_idxs[1]:
                vprint(f"\tDone.")
                idxs_done[k] = True
            else:
                vprint(f"\tFinished at {d}, need at least one more copy...")

            dest[k]=d
            paths[f"{mnr[0][k]}->{mnr[1][k]}"].append(po)
        
        if not all(idxs_done):
            # prepare for another copy of the triangle table
            mnr_tbl = tf_tbl_trig_slice(np.copy(tf_tbl), dest[-1])
            num_copies += 1
            cur_mnr[0,:] = np.array(dest) # the sources of the paths for the next copy are [dest[0],...,dest[-1]]
        else:
            done = True
        if num_copies >= MAX_copies:
            print(f"Reached {MAX_copies} copies, aborting...")
            done = True

    return paths, num_copies  # note that num_copies = len(paths[k]), for any key k.


'''
Classes
=======
'''
class Trig_G():
    '''
    Triangle graph class.
    '''
    def __init__(self, info: list=None) -> None:
        self.info = info
        self.tf_tbl = init_tri_tbl(self.info)
        self.n = len(self.info)+1
        self.valid_ITN = is_L_info_valid_ITN(self.info)
        
    def show_me(self, pre: str='') -> None:
        view_tri_tbl(self.tf_tbl, pre) if self.info else None
        
    def draw_me(self, ax, nd_offset: list=[0.2, 0.3]) -> None:
        draw_triag_graph(self.tf_tbl, ax, noffset=nd_offset)
        
    def __repr__(self):
        return f"Trig_G({self.info})"
    
    def __str__(self):
        return f"Triangle graph representing {self.info} and stored as:\n{self.tf_tbl}\n"
    
    def __len__(self):
        return self.n
    
    def __add__(self, other):
        '''Adding two Trig_G objects implies concatenating uniquely indexes in info.'''
        if self.n == other.n:
            return Trig_G([list(set(self.info[i]+other.info[i])) for i in range(self.n-1)])
        print(f"Adding two Trig_G objects with different sizes is not allowed!!")
            

class LLC_families():
    '''Families of vertex-disjoint paths class based along a triangle graph using the Trig_G class'''
    def __init__(self, t_graph, name: str) -> None:
        self.t_graph = t_graph
        self.indexes = [i+1 for i in range(self.t_graph.n-1)]
        self.name = name
        self.vd_families = None
        self.rvd_families = None
        self.num_copies = None
        self.r = 0
        
    def __repr__(self):
        return f"LLC_families({self.t_graph}, {self.name})"
        
    def display_raw_results(self) -> None:
        print(f"\nResults:\n-----------------")
        print("Number of copies:")
        pprint.pprint(dict(self.num_copies))
        print("Paths:")
        pprint.pprint(dict(self.vd_families))
        print("Reversed paths:")
        pprint.pprint(dict(self.rvd_families))
        print(f"Minimum number of copies to realize all families = {self.r}")

    def display_results(self, reverse=None, compress=None) -> None:
        '''Displays the results.'''
        rsl = self.vd_families if reverse is None else self.rvd_families
        #print(f"\nResults:\n-----------------")
        print(f"{self.name}:")
        for mr, m in zip(rsl.keys(),self.num_copies.keys()):
            print(f"\t{mr}", end=' ')
            print(f"({self.num_copies[m]} copies):") # number of copies
            for p in rsl[mr].keys():
                s = ''.join([f"[{show_1dpath(path_2to1(q), compress=compress)}]" for q in rsl[mr][p]])
                print(f"\t\t{s}")
        print(f"Minimum number of copies to realize all families = {self.r}.")

    def reverse_paths(self) -> None:
        '''This function reverse the paths to be from destination sets to source sets.'''
        rsl = self.vd_families
        Ursl = defaultdict(dict)
        for k in rsl.keys():  # keys here are the minors
            '''
            we reverse the order here since in a upper-right corner minor
            we start the paths with the maximal index (as oppose to the case of
            a lower-left corner minor where we start with the minimal index).
            '''
            Ursl[flip_str_by_sp(k, sp='|')] = \
            {flip_str_by_sp(pi, sp='->'): [p[::-1] for p in rsl[k][pi][::-1]] for pi in sorted(rsl[k].keys(), reverse=True)}
        self.rvd_families = dict(Ursl)


    def run(self, verbose: bool=True) -> None:
        '''This function computes the paths and number of copies.'''
        if verbose:
            print(f'Triangle table:') 
            view_tri_tbl(self.t_graph.tf_tbl)
                
        # containers per minor
        mnrs_num_copies = defaultdict(int)
        mnrs_paths = defaultdict(dict)
        #ll_mnr_indx = [i+1 for i in range(self.t_graph.n-1)]
        # Path analysis for a minor 
        for l in [i+1 for i in range(self.t_graph.n-1)]:
            mnr_tbl = np.copy(self.t_graph.tf_tbl)
            mnr = gen_LL_minor_indexes(self.t_graph.n, l)
            mnr_str = minor_indexes2str(mnr)
            
            # computes paths per minor
            paths, ncopies = run_ll_minor(mnr, mnr_tbl)
                
            if verbose:
                print(f"\n{mnr_str}: {ncopies} copies. Paths:")
                for k in paths.keys():
                    print(f"{k}:")
                    for i in range(len(paths[k])):
                        print(f"\tCopy {i+1}: {paths[k][i]}")  
            mnrs_num_copies[mnr_str] = ncopies
            mnrs_paths[mnr_str] = dict(paths)
                
        self.vd_families = dict(mnrs_paths)
        self.num_copies = dict(mnrs_num_copies)
        self.reverse_paths()  # this populates self.rvd_families
        self.r = max(self.num_copies.values())
                        
    def get_results(self):
        return self.num_copies, self.vd_families, self.rvd_families

class OSC_exp():
    '''Oscillatory matrix class that generates the paths and number of copys for L and U^T (given
    an oscillatory matrix A=LDU), and thus the exponent value.'''
    names: dict = {'L': 'L', 'UT': 'U^T'}

    def __init__(self, A, names=None) -> None:
        if names is not None:
            if len(names) != len(self.names):
                raise ValueError(f"OSC_exp Error: names input must be a list containing {len(self.names)} strings!!")
            self.names = {k:v for k,v in zip(self.names.keys(), names)}
        self.A = A
        self.G: dict = {}  # Trig_G objects for L and for U^T
        self.LC: dict = {} # LLC_families objects for L and U^T
        self.LU_info: dict = {}  # L_i and (U^T)_i with positive parameter values

    def __repr__(self):
        return f"OSC_exp({self.A})"
    
    def __str__(self):
        return f"Paths and exponent of the matrix A=\n{self.A}\n"
    
    def __len__(self):
        return self.A.shape[0]
            
    @classmethod
    def from_vals_ldu(cls, valsl, valsd, valsu):
        '''creating the oscillatory matrix from the SEB factorization parameters'''
        return cls(tpf.compute_matrix_from_EB_factorization(valsl, valsd, valsu))

    def run(self, verbose: bool=False) -> None:
        _, _, _, _, valsL, valsU  = tpf.EB_factorization_ITN( self.A )  
        self.LU_info[self.names['L']] = valsL2L_info(valsL)
        self.LU_info[self.names['UT']] = valsL2L_info(valsU)
        self.G = {self.names['L']: [], self.names['UT']: []}
        self.LC = {self.names['L']: [], self.names['UT']: []}
        
        for k in self.names.values():
            self.G[k] = Trig_G(info=self.LU_info[k])
            self.LC[k] = LLC_families(self.G[k], k)
            self.LC[k].run(verbose=verbose)
        self.r = max([self.LC[k].r for k in self.names.values()])
        
    def display_results(self, show_also_reverse=None, compress=None) -> None:
        show_reverse = False if show_also_reverse is None else True
        for k in self.names.values():
            print(f"\n{k}:")
            self.G[k].show_me()
            self.LC[k].display_results(compress=compress)
            if show_reverse:
                # display reverse paths and number of copies results for L
                print('\nDestination->Source results:')
                self.LC[k].display_results(compress=compress, reverse=True)
        print(f"\nr(A) = {self.r}.")
             

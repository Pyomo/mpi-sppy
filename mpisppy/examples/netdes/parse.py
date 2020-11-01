# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
''' Parse data from an instance file

    TODO: Currently storing the (dense) adjacency matrix. Cheaper not to.
'''
import numpy as np

def main():
    fname = 'data/network-10-10-L-01.dat'
    data = parse(fname)
    print(data['d'][9])
    
    scen = parse(fname, scenario_ix=9)
    print(scen['d'])

def parse(fname, scenario_ix=None):
    with open(fname, 'r') as f:
        ''' Skip file header '''
        while (not f.readline().startswith('+')):
            continue

        ''' Read basic network data '''
        N = int(f.readline().strip())
        density = float(f.readline().strip())
        ratio = int(f.readline().strip())
        A = _toMatrix(f.readline().strip(), dtype=np.int64)
        c = _toMatrix(f.readline().strip(), dtype=np.float64)
        K = int(f.readline().strip())
        p = _toVector(f.readline().strip())

        if (scenario_ix is not None):
            ''' Read the provided scenario '''
            if (scenario_ix < 0 or scenario_ix >= K):
                raise ValueError('Provided scenario index ({}) could '
                                 'not be found ({} total scenarios)'.format(
                                        scenario_ix, K))
            for k in range(scenario_ix):
                for _ in range(4):
                    f.readline()
            f.readline().strip()
            d = _toMatrix(f.readline().strip(), dtype=np.float64)
            u = _toMatrix(f.readline().strip(), dtype=np.float64)
            b = _toVector(f.readline().strip(), dtype=np.float64)
            p = p[scenario_ix]
        else:
            ''' Read all scenarios '''
            d = [None for _ in range(K)]
            u = [None for _ in range(K)]
            b = [None for _ in range(K)]
            for i in range(K):
                f.readline().strip()
                d[i] = _toMatrix(f.readline().strip(), dtype=np.float64)
                u[i] = _toMatrix(f.readline().strip(), dtype=np.float64)
                b[i] = _toVector(f.readline().strip(), dtype=np.float64)

    ''' Construct edge list '''
    ix, iy = np.where(A>0)
    el = [(ix[i],iy[i]) for i in range(len(ix))]

    data = {
        'N': N,
        'density': density,
        'ratio': ratio,
        'A': A,
        'c': c,
        'K': K,
        'p': p,
        'd': d,
        'u': u,
        'b': b,
        'el': el,
    }
    return data
        
def _toVector(line, dtype=np.float64):
    return np.array(line.split(','), dtype=dtype)

def _toMatrix(line, dtype=np.float64):
    rows = line.split(';')
    for i in range(len(rows)):
        rows[i] = rows[i].split(',')
    return np.array(rows, dtype=dtype)


if __name__=='__main__':
    main()

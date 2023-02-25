import sympy as sym
import numpy as np

def H_SM(N_sites, return_list_of_pauli = False):
    J = sym.symbols('J')
    theta = sym.symbols('theta')
    theta0 = sym.symbols('theta0')
    m = sym.symbols('m')
    m0 = sym.symbols('m0')
    w = sym.symbols('w')
    list_of_symbols = [J, theta, theta0, m, m0, w]

    ## symbolic calculations
    Zs = sym.symbols('Z1:{}'.format(N_sites + 1))
    Xs = sym.symbols('X1:{}'.format(N_sites + 1))
    Ys = sym.symbols('Y1:{}'.format(N_sites + 1))

    inv_dict = {}
    for i in range(N_sites):
        inv_dict[Zs[i]] = ('Z', str(i))
        inv_dict[Xs[i]] = ('X', str(i))
        inv_dict[Ys[i]] = ('Y', str(i))

    H_ZZ = 0.
    for n in range(1, N_sites - 1):
        for l in range(1, n + 1):
            for k in range(l):
                H_ZZ += (Zs[l] * Zs[k])*J*.5

    # print(H_ZZ)

    H_XX = 0.
    H_YY = 0.
    for n in range(N_sites - 1):
        H_XX += (w - (-1.) ** (n + 1) * m * sym.sin(theta) / 2) * (Xs[n] * Xs[n + 1])*.5
        H_YY += (w - (-1.) ** (n + 1) * m * sym.sin(theta) / 2) * (Ys[n] * Ys[n + 1])*.5


    H_Z = 0.
    for n in range(N_sites):
        H_Z += (-1) ** (n + 1) * Zs[n] * m * sym.cos(theta) / 2
        if n < N_sites - 1:
            H_Z -= J * ((n + 1) % 2) * sum(Zs[l] for l in range(n + 1)) / 2
    H_Z = sym.collect(sym.expand(H_Z), Zs)
    if return_list_of_pauli:
        return [list_of_symbols, {'zz': H_ZZ, 'z': H_Z, 'xx': H_XX, 'yy': H_YY}, inv_dict, [Xs, Ys, Zs]]
    return [list_of_symbols, {'zz': H_ZZ, 'z': H_Z, 'xx': H_XX, 'yy': H_YY}, inv_dict]

def H_SM_static_charges(N_sites, return_list_of_pauli = False):
    J = sym.symbols('J')
    theta = sym.symbols('theta')
    m = sym.symbols('m')
    m0 = sym.symbols('m0')
    w = sym.symbols('w')
    single_Z_vorfs = sym.symbols('vorf1:{}'.format(N_sites))
    list_of_symbols = [J, theta, m, m0, w, single_Z_vorfs]

    ## symbolic calculations
    Zs = sym.symbols('Z1:{}'.format(N_sites + 1))
    Xs = sym.symbols('X1:{}'.format(N_sites + 1))
    Ys = sym.symbols('Y1:{}'.format(N_sites + 1))

    inv_dict = {}
    for i in range(N_sites):
        inv_dict[Zs[i]] = ('Z', str(i))
        inv_dict[Xs[i]] = ('X', str(i))
        inv_dict[Ys[i]] = ('Y', str(i))

    H_ZZ = 0.
    for n in range(1, N_sites - 1):
        for l in range(1, n + 1):
            for k in range(l):
                H_ZZ += (Zs[l] * Zs[k])*J*.5

    # print(H_ZZ)

    H_XX = 0.
    H_YY = 0.
    for n in range(N_sites - 1):
        H_XX += (w - (-1.) ** (n + 1) * m * sym.sin(theta) / 2) * (Xs[n] * Xs[n + 1])*.5
        H_YY += (w - (-1.) ** (n + 1) * m * sym.sin(theta) / 2) * (Ys[n] * Ys[n + 1])*.5


    H_Z = 0.
    for n in range(N_sites):
        H_Z += (-1) ** (n + 1) * Zs[n] * m * sym.cos(theta) / 2
        if n < N_sites - 1:
            H_Z += single_Z_vorfs[n] * sum(Zs[l] for l in range(n + 1))
    H_Z = sym.collect(sym.expand(H_Z), Zs)
    if return_list_of_pauli:
        return [list_of_symbols, {'zz': H_ZZ, 'z': H_Z, 'xx': H_XX, 'yy': H_YY}, inv_dict, [Xs, Ys, Zs]]
    return [list_of_symbols, {'zz': H_ZZ, 'z': H_Z, 'xx': H_XX, 'yy': H_YY}, inv_dict]

def H_SM_static_charges_single_sector(N_sites, return_list_of_pauli = False):
    J = sym.symbols('J')
    theta = sym.symbols('theta')
    m = sym.symbols('m')
    m0 = sym.symbols('m0')
    w = sym.symbols('w')
    single_Z_vorfs = sym.symbols('vorf1:{}'.format(N_sites))
    list_of_symbols = [J, theta, m, m0, w, single_Z_vorfs]

    ## symbolic calculations
    Zs = sym.symbols('Z1:{}'.format(N_sites + 1))
    Xs = sym.symbols('X1:{}'.format(N_sites + 1))
    Ys = sym.symbols('Y1:{}'.format(N_sites + 1))

    inv_dict = {}
    for i in range(N_sites):
        inv_dict[Zs[i]] = ('Z', str(i))
        inv_dict[Xs[i]] = ('X', str(i))
        inv_dict[Ys[i]] = ('Y', str(i))

    H_ZZ = 0.
    for n in range(1, N_sites - 1):
        for l in range(1, n + 1):
            for k in range(l):
                H_ZZ += (Zs[l] * Zs[k])*J*.5

    # print(H_ZZ)

    H_XX = 0.
    H_YY = 0.
    for n in range(N_sites - 1):
        H_XX += (w - (-1.) ** (n + 1) * m * sym.sin(theta) / 2) * (Xs[n] * Xs[n + 1])*.5
        H_YY += (w - (-1.) ** (n + 1) * m * sym.sin(theta) / 2) * (Ys[n] * Ys[n + 1])*.5


    H_Z = 0.
    for n in range(N_sites-1):
        H_Z += single_Z_vorfs[n] * sum(Zs[l] for l in range(n + 1))
    H_Z = sym.collect(sym.expand(H_Z), Zs)
    if return_list_of_pauli:
        return [list_of_symbols, {'zz': H_ZZ, 'z': H_Z, 'xx': H_XX, 'yy': H_YY}, inv_dict, [Xs, Ys, Zs]]
    return [list_of_symbols, {'zz': H_ZZ, 'z': H_Z, 'xx': H_XX, 'yy': H_YY}, inv_dict]

def evaluate_single_Z_vorf(J_val, m_plus, m_minus, n, Q=1.):
    return J_val * (-((n + 1) % 2)/2. +Q*np.heaviside(n-m_plus, 1.) -Q*np.heaviside(n-m_minus, 1.))

if __name__ == '__main__':
    N_sites = 4

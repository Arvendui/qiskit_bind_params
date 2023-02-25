from qiskit import *
from qiskit.circuit import Parameter
import sympy as sym
import numpy as np
import time
from qiskit import qpy


import load_hamiltonians as load_hamiltonians
from FT_limits import get_reps_E
from rodeo_core import prepare_circuit

if __name__ == '__main__':
    N_sites = 6
    a, b = 0.3, 0.7

    E_scale = 1.
    energy_range = 16.
    density = 2.
    half_sampling_period = .7
    N_ft = int((energy_range / density)/b)  # int((energy_range/density)/0.5)
    N_ft_preciser = int(2.5*N_ft)
    repeat, E_fix = get_reps_E(a, b, -8., -8. + energy_range, half_sampling_period, N_ft)

    load_SM = load_hamiltonians.H_SM_static_charges(N_sites, return_list_of_pauli=True)
    J, theta, m, m0, w, vorfsZ = load_SM[0]
    commuting_groups = {'Z': (1. * (load_SM[1]['z'] + load_SM[1]['zz'])).args, 'X': load_SM[1]['xx'].args, 'Y': load_SM[1]['yy'].args}

    J_param = Parameter('J')
    theta_param = Parameter('theta')
    m_param = Parameter('m')
    w_param = Parameter('w')
    vorfs_params = [Parameter('vorf{}'.format(k)) for k in range(N_sites - 1)]
    SM_no_subs_params = {J: J_param, sym.sin(theta): theta_param.sin(), sym.cos(theta): theta_param.cos(), m: m_param, w: w_param}
    for k in range(N_sites-1):
        SM_no_subs_params[vorfsZ[k]] = vorfs_params[k]
    params = {'m0': 0.5, 'M': 0, 'T': 100., 'N_sites': N_sites,'label':None, 'cycles_estimate': 1, 'cycles_vev': 6, 'use_physical_dev': False, 'repeat': repeat, 'opt_level': 1, 'plot_ft': True, 'N_bootstrap': 100}

    trotter = (1, 1)
    E_param = Parameter('E')
    ts = [Parameter('t{}'.format(oo)) for oo in range(params['cycles_estimate'])]
    backend = Aer.get_backend('aer_simulator')
    N_shots = int(4e4)
    qiskit_params = [backend, E_param, ts]


    collapse_ansatz = 0
    params['collapse'] = collapse_ansatz

    g_val = 1.4
    m_val = 1.
    theta_val = 0.
    tot_len = 15.*0.273
    params['g'] = g_val
    params['m'] = m_val
    params['theta'] = theta_val
    params['a'] = tot_len/(N_sites - 1.)

    ansatz_q = QuantumRegister(N_sites)
    ansatz = QuantumCircuit(ansatz_q, name='Ansatz')
    s = time.time()



    mapped_circ = prepare_circuit(params, ('FT', 1., trotter), E_scale, ('uniform', params['cycles_estimate']),
                                      qiskit_params, ansatz, commuting_groups, load_SM[2], SM_params=SM_no_subs_params,
                                      opt_lev=1, measure_phys=True)

    print(E_fix)
    Es = E_fix * np.ones((N_ft, 1))

    m_plus = int(N_sites / 2)
    m_minus = int(N_sites / 2) + 1

    dict_of_SM_params_evaled = {SM_no_subs_params[J]: params['g'] ** 2 * params['a'] / 2., theta_param: params['theta'], SM_no_subs_params[m]: params['m'], SM_no_subs_params[w]: 1. / (2. * params['a'])}
    list_of_subs = [(m, m_val), (w, 1. / (2. * params['a'])), (theta, theta_val),(J, g_val ** 2 * params['a'] / 2.)]
    for k in range(N_sites-1):
        vorf_val = load_hamiltonians.evaluate_single_Z_vorf(params['g'] ** 2 * params['a'] / 2., m_plus, m_minus, k, Q=1.)
        dict_of_SM_params_evaled[SM_no_subs_params[vorfsZ[k]]] = vorf_val
        list_of_subs.append((vorfsZ[k], vorf_val))
    local_mapped_circ = mapped_circ.bind_parameters(dict_of_SM_params_evaled)

    with open("circuit.qpy", "wb") as qpy_file_write:
        qpy.dump(local_mapped_circ, qpy_file_write)
    print(local_mapped_circ.parameters)
    local_mapped_circ.bind_parameters([0.1, 0.1])
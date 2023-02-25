from qiskit import *
from qiskit.converters import circuit_to_gate
from qiskit.circuit import Parameter
from qiskit.providers.aer import AerSimulator
import sympy as sym
from core import trotter_already_separated, generate_U_from_list,apply_U
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import scipy as scipy
import scipy.odr as odr
import time
from FT_limits import optimal_FT, FT_obs, get_guesses
import matplotlib.pyplot as plt

def prepare_circuit(params, stat, E_scale, method, qiskit_params, ansatz, H_in_groups, inv_dict, measure_phys = False, print_exact = False, print_expect= False, opt_lev = None, SM_params = None):
    # qiskit_params = [backend, E_param, t_param, N_shots]
    N_sites = params['N_sites']

    ancillas = QuantumRegister(method[1])
    cr = ClassicalRegister(method[1])
    qr = QuantumRegister(N_sites)
    all_qcbits = [ancillas, qr, cr]
    if measure_phys:
        cr_phys = ClassicalRegister(N_sites)
        all_qcbits.append(cr_phys)
    if params['collapse'] > 0:
        dummyq = QuantumRegister(method[1])
        dummyc = ClassicalRegister(method[1])
        all_qcbits.append(dummyq)
        all_qcbits.append(dummyc)

    circ = QuantumCircuit(*all_qcbits)

    # apply ansatz state
    if params['collapse'] > 0:
        circ.append(ansatz, [*qr, *dummyq])
        circ.measure(dummyq, dummyc)
    else:
        circ.append(ansatz, qr)

    # init ancillas
    for ancilla in ancillas:
        circ.x(ancilla)
        circ.h(ancilla)

    # rodeo
    for ripetizione in range(params['repeat']):
        for k, ancilla in enumerate(ancillas):
            alls = trotter_already_separated(H_in_groups, *stat[2], inv_dict, SM_params, overall_param=qiskit_params[2][k]/E_scale)
            list_instr = generate_U_from_list(alls, has_factor_2=False)
            temp_qr = QuantumRegister(N_sites)
            temp_circ = QuantumCircuit(temp_qr)
            apply_U(list_instr, temp_circ, temp_qr)
            #print(temp_circ)
            #print(temp_circ.count_ops())
            gate = circuit_to_gate(temp_circ)  # , label='e^{-i H_{obj}*{}}'.format(vorf))
            controlled_gate = gate.control(1)  # , label='C-U^{}'.format(vorf))
            circ.append(controlled_gate, [ancilla, *qr])

            circ.p(qiskit_params[2][k] * qiskit_params[1]/E_scale, ancilla)

    for ancilla in ancillas:
        circ.h(ancilla)

    circ.measure(ancillas, cr)
    if measure_phys:
        circ.measure(qr, cr_phys)
    # print(circ)
    backend = qiskit_params[0]
    if opt_lev != None:
        mapped_circuit = transpile(circ, backend=backend, optimization_level=opt_lev)
    else:
        mapped_circuit = transpile(circ, backend=backend)
    return mapped_circuit

def run_one(mapped_circuit, dict_params, backend, N_shots):
    print('inside run one')
    print(dict_params)
    print(gfhd)
    qobj = assemble(mapped_circuit.bind_parameters(dict_params), backend=backend, shots=N_shots)
    #print(dict_params)
    #print(fghd)
    job = backend.run(qobj)
    counts = job.result().get_counts()
    return counts

def apply_data(params, Es, stat, E_scale, n_reps, method, ansatz, ansatz_param, H_in_groups, inv_dict, measure_phys=False, print_exact=False, print_expect=False, return_sampling = False, return_vev= False, gs_prob = None, mapped_circ = None):
    if mapped_circ == None:
        E_param = Parameter('E')
        ts = [Parameter('t{}'.format(oo)) for oo in range(method[1])]
        all_qiskit_params = list(ts)
        all_qiskit_params.append(E_param)
        for a_param in ansatz_param[0]:
            all_qiskit_params.append(a_param)

        use_physical_dev = params['use_physical_dev']
        if use_physical_dev == False:
            backend = Aer.get_backend('aer_simulator')
            N_shots = int(2e4)
        else:
            provider = IBMQ.load_account()
            #backend = provider.get_backend('ibmq_manila')
            backend = provider.get_backend('ibm_oslo')
            backend = AerSimulator.from_backend(backend)
            N_shots = int(2e4)

        qiskit_params = [backend, E_param, ts]
        start_p = time.time()
        mapped_circuit = prepare_circuit(params, stat, E_scale, method, qiskit_params, ansatz, H_in_groups, inv_dict, measure_phys=measure_phys, print_exact=print_exact, print_expect=print_expect, opt_lev=params['opt_level'])
        print('Time for preparing circuit: ', time.time()-start_p)
    else:
        print('Have mapped circ: ', mapped_circ)
        print('ansatz_param', ansatz_param)
        mapped_circuit, all_qiskit_params, backend, N_shots = mapped_circ

    count_for_diff_experiments = []

    if stat[0] == 'gaussian':
        all_t = np.random.normal(0., stat[1], (method[1], n_reps, Es.shape[1]))
    elif stat[0] == 'uniform':
        all_t = np.random.uniform(-stat[1], stat[1], (method[1], n_reps, Es.shape[1]))
    elif stat[0] == 'FT':
        all_t = np.zeros((method[1], n_reps, Es.shape[1]))
        for kk in range(Es.shape[1]):
            all_t[:, :, kk] = np.linspace(-stat[1], stat[1], n_reps)
    else:
        raise ValueError('You must pass a valid statistics, you passed ', stat[0])
    #print(all_t)
    exc = ProcessPoolExecutor(max_workers=params['ncores'])
    start = time.time()
    for iter in range(Es.shape[1]):
        all_param_values = np.zeros((method[1]+1+len(ansatz_param[0]), n_reps))
        all_param_values[:method[1], :] = all_t[:, :, iter]
        all_param_values[method[1], :] = Es[:, iter]
        for ff in range(len(ansatz_param[0])):
            all_param_values[method[1] + 1 + ff, :] = ansatz_param[1][ff]

        to_run = []
        for pp in range(n_reps):
            to_run.append([mapped_circuit, dict(zip(all_qiskit_params, all_param_values[:, pp])), backend, N_shots])
            print(mapped_circuit.parameters)
            print(all_qiskit_params)
            mapped_circuit.bind_parameters([0.1, 0.1])
            print(ghfd)
            #print(mapped_circ)
            run_one(mapped_circuit, dict(zip(all_qiskit_params, [0.1, 0.1])), backend, N_shots)
        futures = [exc.submit(run_one, *val) for val in to_run]
        counts = []
        for future in futures:
            count = future.result()
            counts.append(count)
        count_for_diff_experiments.append(counts)
    end = time.time()
    print('Time for executing: ', end - start)
    #print('    --> time for a single experiment:', (end - start)/Es.size)

    if return_sampling == True:
        to_ret = []
        for iter in range(Es.shape[1]):
            temp = []
            counts = count_for_diff_experiments[iter]
            for pp in range(n_reps):
                temp.append([all_t[:, pp, iter], counts[pp]])
            to_ret.append(temp)

        return to_ret
    if return_vev == True:
        to_ret = []
        for iter in range(Es.shape[1]):
            counts = count_for_diff_experiments[iter]
            all_vevs = []
            all_counts = []
            for qq, count_s in enumerate(counts):
                phys_res = {}
                for count in count_s:
                    if (' ' + '1'*method[1]) in count:
                        splitted = count.split(' ')
                        phys_res[splitted[0]] = count_s[count]

                if gs_prob != None:
                    if sum(phys_res.values())/N_shots >= (gs_prob[0] + 1.*gs_prob[1]):
                        #print('scartato', sum(phys_res.values())/N_shots)
                        continue
                #print(sum(phys_res.values()), sum(phys_res.values())/N_shots)
                prob_per_run = []
                for i in range(params['N_sites']):
                    counts0 = 0
                    for count in phys_res:
                        if count[params['N_sites'] - i - 1] == '0':
                            counts0 += phys_res[count]
                    p0 = counts0 / sum(phys_res.values())
                    prob_per_run.append(2. * p0 - 1.)

                prob_per_run = np.array(prob_per_run)

                matrix_elements = prob_per_run

                alternating = np.empty(params['N_sites'])
                alternating[::2] = -1.
                alternating[1::2] = +1.

                vev = np.dot(matrix_elements, alternating) / (2. * params['N_sites'] * params['a'])
                all_vevs.append(vev)
                all_counts.append(sum(phys_res.values()))
                #print(vev, sum(phys_res.values()))

            to_ret.append([np.array(all_vevs), np.array(all_counts)])
        return to_ret

def prob_sym(N_peaks, prob_of_peaks, energy_of_peaks, fixed_E, ll):
    t = sym.symbols('t', real='True')
    P1psi = (prob_of_peaks[0]*sym.cos(ll*(fixed_E-energy_of_peaks[0])*t/2)**2)/(sum(prob_of_peaks[i]*sym.cos(ll*(fixed_E-energy_of_peaks[i])*t/2)**2 for i in range(N_peaks)))
    res_lambd = sym.lambdify(t, P1psi)
    return res_lambd

def f_sym(N_peaks, fixed_E, ll, divide = True):
    if N_peaks > 1:
        diag_expect = sym.symbols('diagVEV1:{}'.format(N_peaks + 1), real = 'True')
        off_diag_expect = sym.symbols('offdiagVEV1:{}'.format(int((N_peaks)*(N_peaks-1)/2)+1), real = 'True')
        t = sym.symbols('t', real = 'True')
        prob_of_peaks = sym.symbols('probofpeaks1:{}'.format(N_peaks+1), real = True)
        energy_of_peaks = sym.symbols('energyofpeaks1:{}'.format(N_peaks+1), real = True)

        res = 0.
        off_diag_counter = 0
        if divide:
            P1psi = sum(prob_of_peaks[i]*sym.cos(ll*(fixed_E-energy_of_peaks[i])*t/2)**2 for i in range(N_peaks))
        else:
            P1psi = 1.
        for i in range(N_peaks):
            res += prob_of_peaks[i] * sym.cos(ll * (fixed_E - energy_of_peaks[i]) * t / 2) ** 2 * diag_expect[i] / P1psi
            for j in range(i+1, N_peaks):
                phi_i = ll*(energy_of_peaks[i]- fixed_E)*t
                phi_j = ll*(energy_of_peaks[j]- fixed_E)*t
                res += 2.*sym.re(sym.sqrt(prob_of_peaks[i])*sym.sqrt(prob_of_peaks[j])*(0.5+0.5*sym.exp(-sym.I*phi_i))*(0.5+0.5*sym.exp(+sym.I*phi_j))*off_diag_expect[off_diag_counter])/P1psi
                off_diag_counter += 1
        res_lambd = sym.lambdify([energy_of_peaks, prob_of_peaks, t, (*diag_expect, *off_diag_expect)], res)
        grad_fit_params = [sym.lambdify([energy_of_peaks, prob_of_peaks, t, (*diag_expect, *off_diag_expect)], sym.diff(res, eee)) for eee in
                           [*diag_expect, *off_diag_expect]]
        return res_lambd, grad_fit_params

    diag_expect = sym.symbols('diagVEV', real='True')
    prob_of_peak = sym.symbols('probofpeaks', real=True)
    energy_of_peak = sym.symbols('energyofpeaks', real=True)
    t = sym.symbols('t', real='True')
    res = 0.
    if divide:
        P1psi = prob_of_peak * sym.cos(ll * (fixed_E - energy_of_peak) * t / 2) ** 2
    else:
        P1psi = 1.
    res = prob_of_peak*sym.cos(ll*(fixed_E-energy_of_peak)*t/2)**2 * diag_expect / P1psi

    res_lambd = sym.lambdify([energy_of_peak, prob_of_peak, t, diag_expect], res)
    grad_fit_params = [sym.lambdify([energy_of_peak, prob_of_peak, t, diag_expect], sym.diff(res, eee)) for eee in
                       [diag_expect]]
    return res_lambd, grad_fit_params

def P1(N_peaks, Efix, repeat):
    probs = sym.symbols('p0:{}'.format(N_peaks), real = 'True')
    ees = sym.symbols('E0:{}'.format(N_peaks), real = 'True')
    t = sym.symbols('t', real = 'True')
    res = 0
    for j in range(N_peaks):
        res+= probs[j]*sym.cos(t*(ees[j]-Efix)*repeat/2.)**2
    return sym.lambdify([t, *ees, *probs], res), [sym.lambdify([t, *ees, *probs], sym.diff(res, fitvar)) for fitvar in [*ees, *probs]]

def build_grad(list_of_dervs, t, fit_params):
    res = np.zeros((len(fit_params), t.size))
    for i in range(len(fit_params)):
        res[i, :] = list_of_dervs[i](t, fit_params)
    return res

def build_grad_curve_fit(list_of_dervs, ees, overlaps, t, fit_params):
    res = np.zeros((t.size,len(fit_params)))
    for i in range(len(fit_params)):
        res[:,i] = list_of_dervs[i](ees, overlaps, t, fit_params)
    return res

def fit_data(Npeaks, func, jac, t, y, yerr, beta0, bounds):
    naive_res = scipy.optimize.curve_fit(func, t, y, beta0, jac=jac, sigma=yerr, bounds=bounds, full_output=True)
    ees = naive_res[0][:Npeaks]
    overlaps = naive_res[0][Npeaks:]

    pcov = np.sqrt(np.diag(naive_res[1]))
    ees_inc = pcov[:Npeaks]
    overlaps_inc = pcov[:Npeaks]

    return ees, ees_inc, overlaps, overlaps_inc, naive_res[-1]


def run_experiment(Es, stat, E_scale, max_eigenenergies, params, ansatz, H_in_groups, inv_dict, mapped_circ = None, target=None):
    res_vev = np.zeros((Es.shape[1], 2)) # vev, vev_inc
    all_runs = apply_data(params, Es, stat, E_scale, Es.shape[0], ('uniform', params['cycles_estimate']), ansatz, [[], []], H_in_groups, inv_dict, print_exact=True, return_sampling=True, measure_phys=True, mapped_circ = mapped_circ)
    #print(all_runs)

    all_gs = []
    all_overlap = []
    for out_iter in range(Es.shape[1]):
        x = np.zeros((2+params['cycles_estimate'], Es.shape[0]))
        y = np.zeros(Es.shape[0])
        y_inc = np.ones_like(y)
        vevs = np.zeros(Es.shape[0])
        vevs_inc = np.zeros(Es.shape[0])
        qminus = np.zeros(Es.shape[0])
        qminus_inc = np.zeros(Es.shape[0])
        qtot = np.zeros(Es.shape[0])
        qtot_inc = np.zeros(Es.shape[0])

        for i, res_run in enumerate(all_runs[out_iter]):
            #if res_run[0] == 0.:
                #print('res run', res_run)
            x[0, i] = Es[i, out_iter]
            x[1, i] = E_scale
            x[2:, i] = res_run[0]

            vev_only = {}
            N_counts = 0
            for meas_label in res_run[1]:
                if params['collapse']==0 or '1 ' in meas_label:
                    N_counts += res_run[1][meas_label]
                    if ' 1' in meas_label:
                        y[i] += res_run[1][meas_label]
                        vev_only[meas_label.split(' ')[0+int(params['collapse']>0)]] = res_run[1][meas_label]
            #print(N_counts)
            y_inc[i] = np.sqrt(y[i]+1.)/N_counts#np.sqrt(y[i]/N_counts**2+ y[i]**2/N_counts**3)
            y[i] /= N_counts

            all_probs = np.zeros(params['N_sites'])
            all_inc = np.zeros(params['N_sites'])
            for w in range(params['N_sites']):
                counts0 = 0
                for count in vev_only:
                    if count[params['N_sites'] - w - 1] == '0':
                        counts0 += vev_only[count]
                if sum(vev_only.values()) > 0:
                    p0 = counts0 / sum(vev_only.values())
                    p0_inc = np.sqrt(counts0) / sum(vev_only.values())
                    all_probs[w] = 2. * p0 - 1.
                    all_inc[w] = 2.*p0_inc
                else:
                    all_inc[w] = 1.

            alternating = np.zeros(params['N_sites'])
            alternating[::2] = -1.
            alternating[1::2] = +1.
            first_half = np.zeros(params['N_sites'])
            first_half[:int(params['N_sites']/2)] = 1.
            #print(alternating)
            #print(fdghj)
            vevs[i] = np.dot(all_probs, alternating)/(2 * params['N_sites'])#(2 * params['N_sites'] * params['a'])
            vevs_inc[i] = np.sqrt(np.sum(all_inc**2))/(2 * params['N_sites'])#(2 * params['N_sites'] * params['a'])

            qminus[i] = np.dot(all_probs, first_half)/2.
            qminus_inc[i] = np.sqrt(np.sum((all_inc*first_half) ** 2))/2.

            qtot[i] = np.sum(all_probs)/2.
            qtot_inc[i] = np.sqrt(np.sum(all_inc ** 2))/2.

        '''print(vevs)
        with open('xSMm2r4.npy', 'wb') as f:
            np.save(f, x)
        with open('ySMm2r4.npy', 'wb') as f:
            np.save(f, y)
        with open('vevSMm2r4.npy', 'wb') as f:
            np.save(f, vevs)'''
        #with open('newFT.npy', 'wb') as f:
        #    np.save(f, x[-1, :])
        #    np.save(f, y)
        #    np.save(f, vevs)
        #    np.save(f, qminus)

        '''if y.size % 2 == 0:
            asymm = y[:int(y.size/2)][::-1] - y[int(y.size/2):]
            asymm_inc = np.sqrt(y_inc[:int(y.size/2)][::-1]**2 + y_inc[int(y.size/2):]**2)
            plt.errorbar(x[-1, int(y.size/2):], asymm, yerr=asymm_inc, capsize=4, linestyle = 'none', marker='.')
            plt.grid()
            plt.show()'''

        ees, ees_inc, overlaps,overlaps_inc, tot_prob = optimal_FT(x[-1, :], y,y_inc, x[0, 0], params['repeat'], params['FT_cutoff'], plot=params['plot_ft'])
        # check that overlaps in [0,1]
        for i in range(overlaps.size):
            overlaps[i] = max(0., min(1., overlaps[i]))
        P1_sym, list_of_derivatives = P1(ees_inc.size, x[0,0], params['repeat'])
        def P1_sym_jac(t,*params):
            res = np.zeros((t.size, len(params)))
            for iter, time in enumerate(t):
                res[iter,:] = [evaled(time, *params) for evaled in list_of_derivatives]
            return res
        lower_b = np.zeros(2*ees.size)
        upper_b = np.zeros(2*ees.size)
        lower_b[:ees.size] = -np.inf
        lower_b[ees.size:] = 0.
        upper_b[:ees.size] = np.inf
        upper_b[ees.size:] = 1.
        ees, ees_inc, overlaps, overlaps_inc, flag_energies = fit_data(ees.size, P1_sym, P1_sym_jac, x[-1, :], y, y_inc, np.array([*ees, *overlaps]), (lower_b, upper_b))
        if params['label'] != None:
            print('----------->', params['label'])
        if target != None:
            print('--- Target: {} ----'.format(target))
        print('Eigenenergies of eigenstates in Ansatz are: ', ees, '+/-', ees_inc)
        print('    < n | Ansatz > = ', overlaps, '+/-', overlaps_inc)
        print('    tot. prob:', np.sum(overlaps), '+/-', np.sqrt(np.sum(overlaps_inc**2)))
        print('    flag:', flag_energies)
        if ees.size > 1:
            prob_f = prob_sym(ees.size, overlaps, ees, x[0, 0], 1.)
            samplings = np.linspace(-3., 3., 5000)#np.linspace(-stat[1], stat[1], Es.shape[0])
            samplings = samplings[5:-5]
            results = prob_f(samplings)
            best_index = np.argmax(results)
            #print('"Optimal time:" ', samplings[best_index], 'prob of not being in max peak: ', overlaps[0]*np.cos(1.*(x[0,0]-ees[0])*samplings[best_index]/2)**2/results[best_index], 'prob of gs:', overlaps[0]*np.cos(1.*(x[0,0]-ees[0])*samplings[best_index]/2)**2)
        # compute vev
        N_peaks = ees.size
        f_sym_ev, grad_sym = f_sym(ees.size, x[0, 0], params['repeat'])
        f_sym_fit = lambda t, *params_l: f_sym_ev(list(ees), list(overlaps), t, params_l)
        # f_sym_ev_noP, grad_sym_noP = f_sym(ees.size, overlaps, ees, x[0, 0], params['repeat'], divide=False)
        # f_sym_fit_noP = lambda t, *params_l: f_sym_ev_noP(t, params_l)
        if N_peaks > 1:
            lower_b = np.ones(N_peaks + int((N_peaks) * (N_peaks - 1) / 2))
            upper_b = np.ones(N_peaks + int((N_peaks) * (N_peaks - 1) / 2))
            lower_b[:N_peaks] = -1. / (2.)
            upper_b[:N_peaks] = 1. / (2.)
            lower_b[N_peaks:] = -1. / (2.)
            upper_b[N_peaks:] = 1. / (2.)

            # for qminus
            lower_bq = np.ones(N_peaks + int((N_peaks) * (N_peaks - 1) / 2))
            upper_bq = np.ones(N_peaks + int((N_peaks) * (N_peaks - 1) / 2))
            lower_bq[:N_peaks] = -int(params['N_sites'] / 2) / (2.)
            upper_bq[:N_peaks] = int(params['N_sites'] / 2) / (2.)
            lower_bq[N_peaks:] = -1. * int(params['N_sites'] / 2) / (2.)
            upper_bq[N_peaks:] = 1. * int(params['N_sites'] / 2) / (2.)

            # for qtot
            lower_bqt = np.ones(N_peaks + int((N_peaks) * (N_peaks - 1) / 2))
            upper_bqt = np.ones(N_peaks + int((N_peaks) * (N_peaks - 1) / 2))
            lower_bqt[:N_peaks] = -int(params['N_sites']) / (2.)
            upper_bqt[:N_peaks] = int(params['N_sites']) / (2.)
            lower_bqt[N_peaks:] = -1. * int(params['N_sites']) / (2.)
            upper_bqt[N_peaks:] = 1. * int(params['N_sites']) / (2.)


            beta0_vev = get_guesses(ees.size,*FT_obs(ees, overlaps, x[-1, :], vevs, x[0, 0], params['repeat']), -0.5, 0.5)
            beta0_qminus = get_guesses(ees.size,*FT_obs(ees, overlaps, x[-1, :], qminus, x[0, 0], params['repeat']),-int(params['N_sites'] / 2) / 2., int(params['N_sites'] / 2) / 2.)
            beta0_qtot = get_guesses(ees.size, *FT_obs(ees, overlaps, x[-1, :], qminus, x[0, 0], params['repeat']), -int(params['N_sites']) / 2., int(params['N_sites']) / 2.)

            fit_res_vev = scipy.optimize.curve_fit(f_sym_fit, x[-1, :], vevs, sigma=vevs_inc,p0=beta0_vev, bounds=(lower_b, upper_b),jac=lambda t, *params_l: build_grad_curve_fit(grad_sym,list(ees),list(overlaps), t,params_l),full_output=True)
            popt_vev, pcov_vev = fit_res_vev[0], fit_res_vev[1]
            vevs_val = popt_vev[:N_peaks]
            vevs_val_inc = np.sqrt(np.diag(pcov_vev))[:N_peaks]
            print('scipy vev: ', vevs_val[0], vevs_val_inc[0],'----------->vev flag:', fit_res_vev[-2:])

            fit_res_qminus = scipy.optimize.curve_fit(f_sym_fit, x[-1, :], qminus, sigma=qminus_inc,
                                                      p0=beta0_qminus, bounds=(lower_bq, upper_bq),
                                                      jac=lambda t, *params_l: build_grad_curve_fit(grad_sym,
                                                                                                    list(ees),
                                                                                                    list(overlaps),
                                                                                                    t,
                                                                                                    params_l),
                                                      full_output=True)
            popt_qminus, pcov_qminus = fit_res_qminus[0], fit_res_qminus[1]
            qminus_valS = popt_qminus[:N_peaks]
            qminus_valS_inc = np.sqrt(np.diag(pcov_qminus))[:N_peaks]

            print('scipy qminus: ', qminus_valS[0], qminus_valS_inc[0],'----------->qminus flag:', fit_res_qminus[-2:])

            fit_res_qtot = scipy.optimize.curve_fit(f_sym_fit, x[-1, :], qtot, sigma=qtot_inc,
                                                      p0=beta0_qtot, bounds=(lower_bqt, upper_bqt),
                                                      jac=lambda t, *params_l: build_grad_curve_fit(grad_sym,
                                                                                                    list(ees),
                                                                                                    list(overlaps),
                                                                                                    t,
                                                                                                    params_l),
                                                      full_output=True)
            popt_qtot, pcov_qtot = fit_res_qtot[0], fit_res_qtot[1]
            qtot_valS = popt_qtot[:N_peaks]
            qtot_valS_inc = np.sqrt(np.diag(pcov_qtot))[:N_peaks]

            print('scipy qtot: ', qtot_valS[0], qtot_valS_inc[0], '----------->qminus flag:', fit_res_qtot[-2:])
            assert Es.shape[1] == 1
            return (ees, ees_inc, overlaps, overlaps_inc, vevs_val, vevs_val_inc, qminus_valS, qminus_valS_inc, qtot_valS, qtot_valS_inc)
        else:
            f_sym_ev, grad_sym = f_sym(ees.size, x[0, 0], params['repeat'])
            def f_sym_fit(t, param_l):
                res = np.zeros_like(t)
                for it, time in enumerate(t):
                    res[it] = f_sym_ev(ees, overlaps, time, param_l)
                return res
            beta0 = np.zeros(1)
            lower_b = -1./2.
            upper_b = 1./2.

            beta0_vev = get_guesses(ees.size, *FT_obs(ees, overlaps, x[-1, :], vevs, x[0, 0], params['repeat']), -0.5,
                                    0.5)
            beta0_qminus = get_guesses(ees.size, *FT_obs(ees, overlaps, x[-1, :], qminus, x[0, 0], params['repeat']),
                                       -int(params['N_sites'] / 2) / 2., int(params['N_sites'] / 2) / 2.)
            beta0_qtot = get_guesses(ees.size, *FT_obs(ees, overlaps, x[-1, :], qminus, x[0, 0], params['repeat']), -int(params['N_sites']) / 2., int(params['N_sites']) / 2.)


            fit_res_vev = scipy.optimize.curve_fit(f_sym_fit, x[-1, :], vevs, p0=beta0_vev, sigma=vevs_inc,bounds=(lower_b, upper_b), full_output=True)
            vevs_val = fit_res_vev[0]
            vevs_val_inc = np.sqrt(fit_res_vev[1])
            print('scipy vev, ', vevs_val[0], vevs_val_inc[0])

            fit_res_qminus = scipy.optimize.curve_fit(f_sym_fit, x[-1, :], qminus, p0=beta0_qminus, sigma=qminus_inc,bounds=(-1. * int(params['N_sites'] / 2) / (2.), 1. * int(params['N_sites'] / 2) / (2.)), full_output=True)
            print('scipy qminus', fit_res_qminus[0][0], np.sqrt(fit_res_qminus[1])[0])

            fit_res_qtot = scipy.optimize.curve_fit(f_sym_fit, x[-1, :], qtot, p0=beta0_qtot, sigma=qtot_inc,
                                                      bounds=(-1. * int(params['N_sites']) / (2.),
                                                              1. * int(params['N_sites']) / (2.)), full_output=True)
            print('scipy qtot', fit_res_qtot[0][0], np.sqrt(fit_res_qtot[1])[0])

            assert Es.shape[1] == 1
            return (ees, ees_inc, overlaps, overlaps_inc, vevs_val, vevs_val_inc, fit_res_qminus[0], np.sqrt(fit_res_qminus[1]), fit_res_qtot[0], np.sqrt(fit_res_qtot[1]))

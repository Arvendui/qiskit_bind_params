import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal.windows import blackman, flattop, hann
from scipy.signal import find_peaks, peak_widths
import matplotlib.pyplot as plt

def get_N_E(T_tot, Emin, Emax):
    A_inv = np.array([[1/(2*np.pi), -1/(2*np.pi)], [-0.1, 0.45]])*2*np.pi/0.35
    b = np.array([[Emax/(2*np.pi) - 0.9], [Emin/(2*np.pi) -0.2]])

    return np.dot(A_inv, b)

def get_T_E(Emin, Emax, ll, N):
    T = 0.5/2*(N-2)*2*np.pi/(ll*(Emax-Emin))
    Efix = Emin - 0.25/2*(N-2)/T * 2*np.pi/ll
    # print('T/2 is', T/2, '---> {} pi'.format(T/(2*np.pi)))
    # print('Efix', Efix)
    return (T/2, Efix)

def get_reps_E(a, b, Emin, Emax, Thalfs, N):
    reps = max(int(((b/2*(N-2)*2*np.pi)/(2.*Thalfs))/(Emax-Emin)), 1)
    Efix = Emin - a/2*(N-2)/(2.*Thalfs) * 2*np.pi/reps
    return (reps, Efix)

def optimal_FT(t, y,yinc, Efix,repeat, FT_cutoff, cutoffs = None, plot = False):
    if cutoffs == None:
        return optimal_FT(t, y, yinc, Efix, repeat, FT_cutoff, cutoffs = [1.], plot = plot)
    ees = []
    ees_inc = []
    overlaps = []
    overlaps_inc = []
    tots = []
    cutoffs = np.array(cutoffs)*np.amax(t)
    for cutoff in cutoffs:
        relevant_indices = np.where(np.abs(t) <= cutoff)[0]
        relevant_t = t[relevant_indices]
        relevant_y = y[relevant_indices]
        relevant_y_inc = yinc[relevant_indices]

        Tmin, Tmax = np.amin(relevant_t), np.amax(relevant_t)
        omegas = rfftfreq(relevant_y.size, (Tmax - Tmin) / relevant_y.size)


        windows = [blackman(t.size), hann(t.size)]#, np.ones(t.size)]

        ft_res = []
        ft_res_inc = []
        for w in range(len(windows)):
            ft_res.append(rfft(relevant_y * windows[w])*2./np.sum(windows[w]))
            ft_res_inc.append(rfft(relevant_y_inc * windows[w]) * 2. / np.sum(windows[w]))

        ft_peaks = list(map(lambda index: find_peaks(np.abs(ft_res[index]), height=FT_cutoff), range(len(ft_res)))) #height=np.median(np.abs(ft_res[index]))+4.*np.median(np.absolute(np.abs(ft_res[index]) - np.median(np.abs(ft_res[index]))))), range(len(ft_res))))
        most_peaks = np.argmax(list(map(lambda v: v[0].size, ft_peaks)))

        #print(list(map(lambda v: v[0].size, ft_peaks)), most_peaks)

        peaks_widths = peak_widths(np.abs(ft_res[most_peaks]), ft_peaks[most_peaks][0], rel_height=.15)

        #print('(raw) pos of peaks:', omegas[ft_peaks[most_peaks][0]])
        #print('with widths', peaks_widths)

        if ft_peaks[most_peaks][0].size > 0:
            ees.append(Efix + np.pi * omegas[ft_peaks[most_peaks][0]] / (repeat / 2.))
            ees_inc.append(np.pi * peaks_widths[0]*0.5 / (repeat / 2.))
            overlaps.append(2. * np.abs(ft_res[most_peaks])[ft_peaks[most_peaks][0]])
            overlaps_inc.append((2. * np.abs(ft_res_inc[most_peaks])[ft_peaks[most_peaks][0]]))
            tots.append(np.sum(overlaps[-1]))

            if plot:
                fig, axs = plt.subplots(2, gridspec_kw={'height_ratios': [1, 2]})
                axs[0].errorbar(relevant_t, relevant_y, yerr = relevant_y_inc, linestyle=':', linewidth = 1, marker='.', capsize = 4)
                axs[0].grid()
                axs[1].grid()
                axs[0].set_ylim(0., 1.)
                axs[0].set_xlabel('t')
                axs[0].set_ylabel('prob.')
                axs[1].errorbar(omegas, np.abs(ft_res[most_peaks]), yerr=np.abs(ft_res_inc[most_peaks]),linestyle=':',linewidth = 1.2, marker='.', capsize = 4, color ='red')
                axs[1].plot(omegas[ft_peaks[most_peaks][0]], np.abs(ft_res[most_peaks])[ft_peaks[most_peaks][0]], marker='o', linestyle='none')
                #axs[1].plot([omegas[0], omegas[-1]], [np.mean(np.abs(ft_res[most_peaks])),np.mean(np.abs(ft_res[most_peaks]))], color ='black', linestyle = ':')
                #axs[1].plot([omegas[0], omegas[-1]],[np.median(np.abs(ft_res[most_peaks])), np.median(np.abs(ft_res[most_peaks]))], color='black',linestyle='-.')
                #axs[1].plot([omegas[0], omegas[-1]],[np.median(np.abs(ft_res[most_peaks]))+np.median(np.absolute(np.abs(ft_res[most_peaks]) - np.median(np.abs(ft_res[most_peaks])))), np.median(np.abs(ft_res[most_peaks]))+np.median(np.absolute(np.abs(ft_res[most_peaks]) - np.median(np.abs(ft_res[most_peaks]))))],color='grey', linestyle='-.')
                axs[1].plot([omegas[0], omegas[-1]], [np.median(np.abs(ft_res[most_peaks])) + 4.*np.median(
                    np.absolute(np.abs(ft_res[most_peaks]) - np.median(np.abs(ft_res[most_peaks])))),
                                                      np.median(np.abs(ft_res[most_peaks])) + 4.*np.median(np.absolute(
                                                          np.abs(ft_res[most_peaks]) - np.median(
                                                              np.abs(ft_res[most_peaks]))))], color='cyan',
                            linestyle='-.')
                axs[1].set_yscale('log')
                plt.show()
        else:
            if plot:
                fig, axs = plt.subplots(2, gridspec_kw={'height_ratios': [2, 1]})
                axs[0].errorbar(relevant_t, relevant_y, yerr=relevant_y_inc, linestyle=':', linewidth=1, marker='.',capsize=4)
                axs[0].grid()
                axs[1].grid()
                axs[0].set_ylim(0., 1.)
                axs[0].set_xlabel('t')
                axs[0].set_ylabel('prob.')
                axs[1].errorbar(omegas, np.abs(ft_res[most_peaks]), yerr = np.abs(ft_res_inc[most_peaks]), linestyle=':', linewidth=0.5, marker='.',
                                capsize=4, color='red')
                axs[1].set_yscale('log')
                plt.show()
    if len(tots) > 0:
        tots = np.array(tots)
        return (ees[np.argmax(tots)], ees_inc[np.argmax(tots)], overlaps[np.argmax(tots)], overlaps_inc[np.argmax(tots)], tots[np.argmax(tots)])
    return None

def to_cc(tupla_norm_angle):
    return tupla_norm_angle[0]*np.exp(1.j*tupla_norm_angle[1])

def FT_obs(ees, overlaps, t, observable, Efix, repeat, plot = False):
    window = blackman(t.size)
    omegas = rfftfreq(t.size, (np.amax(t) - np.amin(t)) / t.size)
    ft = rfft(observable*window)*2./np.sum(window)

    # diag peaks
    diag_omega = (ees-Efix)*repeat/(2.*np.pi)

    diag_omega_closest = {}
    for i in range(diag_omega.size):
        diag_omega_closest[i] = np.argmin(np.abs(omegas-diag_omega[i]))
    #print(diag_omega_closest)
    off_diag_omega = []
    off_diag_omega_closest = {}
    for j in range(ees.size):
        for l in range(j+1, ees.size):
            off_diag_omega.append((ees[l]-ees[j])*repeat/(2*np.pi))
            off_diag_omega_closest[(j,l)] = np.argmin(np.abs(omegas - off_diag_omega[-1]))
    off_diag_omega = np.array(off_diag_omega)

    #print(off_diag_omega_closest)

    decomposed_diag = {}
    for key in diag_omega_closest:
        #print(ft[diag_omega_closest[key]])
        decomposed_diag[key] = (np.abs(ft[diag_omega_closest[key]]), np.angle(ft[diag_omega_closest[key]]))

    decomposed_off_diag = {}
    for key in off_diag_omega_closest:
        #print(ft[off_diag_omega_closest[key]])
        decomposed_off_diag[key] = (np.abs(ft[off_diag_omega_closest[key]]), np.angle(ft[off_diag_omega_closest[key]]))

    #print(decomposed_diag)
    #print(decomposed_off_diag)
    #print(gfhd)

    betas = np.zeros((ees.size, ees.size))
    vevs = np.zeros((ees.size, ees.size))
    for key in decomposed_off_diag:
        vevs[key] = decomposed_off_diag[key][0]*2./(np.sqrt(overlaps[key[0]])*np.sqrt(overlaps[key[1]]))
        betas[key] = decomposed_off_diag[key][1]

    #print(vevs)
    #print(betas)

    for key in decomposed_diag:
        tot_value = to_cc(decomposed_diag[key])
        for l in range(key + 1, ees.size):
            tot_value -= 0.5*np.sqrt(overlaps[key])*np.sqrt(overlaps[l])*np.exp(-1.j*betas[key, l])*vevs[key, l]

        vevs[key, key] = 2.*np.abs(tot_value)
        betas[key, key] = np.angle(tot_value)

    if plot:
        fig, axs = plt.subplots(2, gridspec_kw={'height_ratios': [1, 2]})
        axs[0].plot(t, observable, #yerr=observavle_inc, capsize=4,
                         linestyle=':', linewidth=1, marker='.')
        axs[0].grid()
        axs[1].grid()
        axs[0].set_xlabel('t')
        axs[0].set_ylabel('obs.')
        axs[1].plot(omegas, np.abs(ft), linestyle = ':', color = 'black', linewidth= 1.5)
        axs[1].plot(omegas, np.real(ft), #yerr=np.abs(ft_res_inc[most_peaks]),capsize=4,
                     linestyle=':', linewidth=.75,  color='red')
        axs[1].plot(omegas, np.imag(ft),  # yerr=np.abs(ft_res_inc[most_peaks]),capsize=4,
                    linestyle=':', linewidth=.75, color='blue')
        axs[1].scatter(diag_omega, np.ones_like(diag_omega)*0.05, color = 'cyan')
        axs[1].scatter(off_diag_omega, np.ones_like(off_diag_omega) * 0.03, color='orange')
        axs[1].set_yscale('linear')
        plt.show()

    return vevs, betas

def get_guesses(N_peaks, obs_estimate, betas, v_min, v_max):
    beta0 = np.zeros(N_peaks + int((N_peaks) * (N_peaks - 1) / 2))
    beta0[:N_peaks] = list(map(lambda x: max(v_min, min(v_max, x)), np.diag(obs_estimate)))

    off_norm = []
    off_phase = []
    for j in range(N_peaks):
        for lc, l in enumerate(range(j + 1, N_peaks)):
            off_norm.append(max(v_min, min(v_max, obs_estimate[j, l])))
            off_phase.append(max(-np.pi, min(np.pi, betas[j, l] - betas[j, j] + betas[l, l])))

    off_norm = np.array(off_norm)
    off_phase = np.array(off_phase)

    beta0[N_peaks:] = np.real(off_norm * np.exp(1.j * off_phase))
    return beta0

if __name__ == '__main__':
    get_T_E(-2., 2., 40, 300)
    get_T_E(-2., 0., 50, 200)
"""
Die Simulation einer sendenden Antenne und einer mobillen empfangenden Antenne,
welche sich zueinander verdrehen (und verschieben, welches aber durch
Verdehungen abgebildet werden kann).
"""
import numpy as np
import matplotlib.pyplot as plt

"""
Funktionsdeklarationen:
"""


def get_distance_2D(x_a, x_b):
    x_ab = x_a - x_b
    dist = ((x_ab[0][0])**2 + (x_ab[1][0])**2)**0.5
    return dist


def get_distance_1D(x_a, x_b):
    dist = abs(x_a - x_b)
    return dist


# Vektor wird auf Ebene projeziert und Winkel mit main-Vektor gebildet
def get_angle_v_on_plane(v_x, v_1main, v_2):
    g_mat = np.array([[np.dot(v_2.T, v_2)[0][0], np.dot(v_2.T, v_1main)[0][0]], [np.dot(v_1main.T, v_2)[0][0], np.dot(v_1main.T, v_1main)[0][0]]])
    g_vec = np.array([[np.dot(v_x.T, v_2)[0][0]], [np.dot(v_x.T, v_1main)[0][0]]])
    gamma_x = np.linalg.solve(g_mat, g_vec)
    v_x_proj = gamma_x[0]*v_2 + gamma_x[1]*v_1main
    angle_x = np.arccos(np.dot(v_x_proj.T, v_1main)[0][0] / (np.linalg.norm(v_x_proj) * np.linalg.norm(v_1main)))
    return angle_x


def get_angles(x_current, tx_pos, h_tx, z_mauv, h_mauv):
    dh = h_mauv - h_tx
    r = x_current - tx_pos
    r_abs = np.linalg.norm(r)
    phi_cap = np.arccos(r[0][0]/r_abs)
    if r[1] <= 0:
        phi_cap = 2*np.pi - phi_cap
    theta_cap = np.arctan(dh / r_abs)
    S_Ktx_KB = np.array([[np.cos(phi_cap)*np.cos(theta_cap), -np.sin(phi_cap), -np.cos(phi_cap)*np.sin(theta_cap)],
                         [np.sin(phi_cap)*np.cos(theta_cap), np.cos(phi_cap), -np.sin(phi_cap)*np.sin(theta_cap)],
                         [np.sin(theta_cap), 0, np.cos(theta_cap)]]).T
    # Verdrehmatrix um z & phi, dann y & theta --- [0]=x_B.T, [1]=y_B.T, [2]=z_B.T
    phi_low = get_angle_v_on_plane(z_mauv, np.array(S_Ktx_KB[2])[np.newaxis].T, np.array(S_Ktx_KB[0])[np.newaxis].T)            # <---------------
    theta_low = get_angle_v_on_plane(z_mauv, np.array(S_Ktx_KB[2])[np.newaxis].T, np.array(S_Ktx_KB[1])[np.newaxis].T)
    return phi_cap, theta_cap, phi_low, theta_low


def h_rss_model(x_pos_mobil, x_pos_stat, alpha, gamma, theta_cap, phi_low, theta_low, n_tx, n_rec):
    r = get_distance_2D(x_pos_mobil, x_pos_stat)
    rss = -20*np.log10(r)-r*alpha + gamma + 10*np.log10((np.cos(phi_low)**2) * (np.cos(theta_cap)**n_tx) * (np.cos(theta_cap + theta_low)**n_rec))
    return rss, r


def h_rss_messungsemulator(x_pos_mobil, x_pos_stat, alpha, gamma, theta_cap, phi_low, theta_low, n_tx, n_rec):
    r = get_distance_2D(x_pos_mobil, x_pos_stat)
    rss = -20*np.log10(r)-r*alpha + gamma  # + 10*np.log10((np.cos(phi_low)**2) * (np.cos(theta_cap)**n_tx) * (np.cos(theta_cap + theta_low)**n_rec))
    return rss


def h_rss_jacobi(x_pos_mobil, x_pos_stat, alpha):
    r = get_distance_2D(x_pos_mobil, x_pos_stat)
    h_rss_jacobimatrix = np.empty([2, 1])
    h_rss_jacobimatrix[0] = -20*(x_pos_mobil[0]-x_pos_stat[0])/(np.log(10)*r**2)-alpha*(x_pos_mobil[0]-x_pos_stat[0])/r
    h_rss_jacobimatrix[1] = -20*(x_pos_mobil[1]-x_pos_stat[1])/(np.log(10)*r**2)-alpha*(x_pos_mobil[1]-x_pos_stat[1])/r
    return h_rss_jacobimatrix


def measurement_covariance_model(rss_noise_model, r_dist, ekf_param_Xtra):
    
    ''' EVTL HIER NOCH ERWEITERUNG DES MODELLS MIT WINKELUNSICHERHEIT '''
    
    ekf_param = [6.5411, 7.5723, 9.5922, 11.8720, 21.6396, 53.6692, 52.0241]
    r_sig = 50.0 + ekf_param_Xtra  # default (Fuer RSS zwischen -35 und -55 z.Z.)
    if -35 < rss_noise_model or r_dist >= 1900:
        r_sig = 100 + ekf_param_Xtra
    else:
        if rss_noise_model >= -55:
            r_sig = ekf_param[0] + ekf_param_Xtra
        elif rss_noise_model < -80:
            r_sig = ekf_param[4] + ekf_param_Xtra
        elif rss_noise_model < -75:
            r_sig = ekf_param[3] + ekf_param_Xtra
        elif rss_noise_model < -65:
            r_sig = ekf_param[2] + ekf_param_Xtra
        elif rss_noise_model < -55:
            r_sig = ekf_param[1] + ekf_param_Xtra
    r_mat = r_sig ** 2
    return r_mat


def ekf_prediction(x_est, p_mat, q_mat):
    x_est = x_est  # + np.random.randn(2, 1) * 1  # = I * x_est
    p_mat = i_mat.dot(p_mat.dot(i_mat)) + q_mat  # Theoretisch mit .T transponierten zweiten I Matrix
    return x_est, p_mat


def ekf_update(rss, tx_pos, tx_alpha, tx_gamma, x_est, p_mat, ekf_param_Xtra, txh, zmauv, hmauv, txn, recn):
    z_meas = rss
    for itx in range(tx_num):
        phi_cap_itx, theta_cap_itx, phi_low_itx, theta_low_itx = get_angles(x_est, tx_pos[itx], txh[itx], zmauv, hmauv)
        y_est, r_dist = h_rss_model(x_est, tx_pos[itx], tx_alpha[itx], tx_gamma[itx], theta_cap_itx, phi_low_itx, theta_low_itx,
                              txn[itx], recn)
        y_tild = z_meas[itx] - y_est
        r_mat = measurement_covariance_model(z_meas[itx], r_dist, ekf_param_Xtra)
        h_jac_mat = h_rss_jacobi(x_est, tx_pos[itx], tx_alpha[itx])
        s_mat = np.dot(h_jac_mat.T, np.dot(p_mat, h_jac_mat)) + r_mat
        k_mat = np.dot(p_mat, (h_jac_mat / s_mat))
        x_est = x_est + np.dot(k_mat, y_tild)
        p_mat = (i_mat - np.dot(k_mat, h_jac_mat.T)) * p_mat
    return x_est, p_mat


"""
Ausfuehrendes Programm:
"""

np.random.seed(12896)

'''Konfiguration der Messpunkte'''
dist_messpunkte = 25.0
start_messpunkte = np.array([[0.0], [0.0]])

x_n = [start_messpunkte]
while x_n[-1][0] < 1000.0:
    start_messpunkte = start_messpunkte + np.array([[dist_messpunkte], [0.0]])
    x_n.append(start_messpunkte)
while x_n[-1][1] < 1000.0:
    start_messpunkte = start_messpunkte + np.array([[0.0], [dist_messpunkte]])
    x_n.append(start_messpunkte)
while x_n[-1][0] > 0.0:
    start_messpunkte = start_messpunkte + np.array([[-dist_messpunkte], [0.0]])
    x_n.append(start_messpunkte)
while x_n[-1][1] > 0.0:
    start_messpunkte = start_messpunkte + np.array([[0.0], [-dist_messpunkte]])
    x_n.append(start_messpunkte)
anz_messpunkte = len(x_n)

'''Konfiguration der Hoehe und der Antennenverdrehung durch Beschreibung des mobilen Antennenvektors'''
h_mauv = 0
z_mauv = np.array([[100], [0], [100]])

'''Bestimmung der Messfrequenzen'''
tx_freq = [4.3400e+08, 4.341e+08, 4.3430e+08, 4.3445e+08, 4.3465e+08, 4.3390e+08]
tx_num = len(tx_freq)

'''Postion(en) der stationaeren Antenne(n)'''
tx_pos = [np.array([[-100.9], [-100.9]]), np.array([[500.9], [-100.9]]), np.array([[1100.9], [-100.9]]),
          np.array([[-100.9], [1100.9]]), np.array([[500.9], [1100.9]]), np.array([[1100.9], [1100.9]])]

'''Kennwerte der stationaeren Antenne(n)'''
tx_alpha = np.array([0.01110, 0.01401, 0.01187, 0.01322, 0.01021, 0.01028])
tx_gamma = np.array([-0.49471, -1.24823, -0.17291, -0.61587, 0.99831, 0.85711])
tx_n = np.array([2, 2, 2, 2, 2, 2])
tx_h = np.array([0, 0, 0, 0, 0, 0])

'''Kennwerte der Rauschenden Abweichungen der Antennen'''
tx_sigma = np.array([0.33, 0.33, 0.33, 0.33, 0.33, 0.33])

'''Kennwerte der mobilen Antenne'''
rec_n = 2

'''Extra Unsicherheit fuer R-Matrix'''
ekf_param_Xtra = 0

'''Initialisierung der P-Matrix (Varianz der Position)'''
p_mat = np.array([[100**2, 0], [0, 100**2]])  # Abweichungen von x1 und x2 aufgrund der Messungen...

'''Initialisierung der Q-Matrix (Varianz des Prozessrauschens / Modellunsicherheit)'''
q_mat = np.array([[500**2, 0], [0, 500**2]])  # Abweichungen von x1 und x2 aufgrund des Modelles

'''Initialisierung der y-Matrix fuer die erwartete Messung'''
y_est = np.zeros(tx_num)

'''Initialisierung der F-Matrix -> Gradienten von f(x)'''
i_mat = np.eye(2)

'''Initialisierung der Distanzspeicherungsmatrix'''
r_dist = np.zeros(tx_num)

'''Initialisierung der Messmatrix'''
z_meas = np.zeros(tx_num)

'''Initialisierung der geschaetzten Position'''
x_est = np.array([[500.0], [500.0]])

'''Initialisierung der Liste(n) fuer Plots'''
x_est_list = [x_est]

for k in range(anz_messpunkte):
    print "\n \n \nDurchlauf Nummer", k

    rss = np.empty(tx_num)
    for i in range(tx_num):  # "Messung" ff.
        phi_cap_t, theta_cap_t, phi_low_t, theta_low_t = get_angles(x_n[k], tx_pos[i], tx_h[i], z_mauv, h_mauv)
        rss[i] = h_rss_messungsemulator(x_n[k], tx_pos[i], tx_alpha[i], tx_gamma[i], theta_cap_t, phi_low_t, theta_low_t, tx_n[i], rec_n) + np.random.randn(1)*tx_sigma[i]
    x_est, p_mat = ekf_prediction(x_est, p_mat, q_mat)
    x_est, p_mat = ekf_update(rss, tx_pos, tx_alpha, tx_gamma, x_est, p_mat, ekf_param_Xtra, tx_h, z_mauv, h_mauv, tx_n, rec_n)

    print "Die wirkliche Position ist: \n", x_n[k]
    print "Die geschaetzte Position ist: \n", x_est
    print "( Die p-Matrix entspricht: \n", p_mat, ")"

    x_est_list.append(x_est)

print('\nFertich!\n')

'''
Plotting
'''

'''Erstellung der X und Y Koordinatenlisten zum einfachen und effizienteren Plotten'''
x_n_x = [None]*len(x_n)
x_n_y = [None]*len(x_n)
x_est_x = [None]*len(x_est_list)
x_est_y = [None]*len(x_est_list)
tx_pos_x = [None]*len(tx_pos)
tx_pos_y = [None]*len(tx_pos)

for i in range(0, len(x_n)):
    x_n_x[i] = x_n[i][0]
    x_n_y[i] = x_n[i][1]
for i in range(0, len(x_est_list)):
    x_est_x[i] = x_est_list[i][0]
    x_est_y[i] = x_est_list[i][1]
for i in range(0, len(tx_pos)):
    tx_pos_x[i] = tx_pos[i][0]
    tx_pos_y[i] = tx_pos[i][1]

'''Strecke im Scatterplot'''
plt.figure(figsize=(25, 12))
plt.subplot(121)
plt.scatter(x_n_x, x_n_y, marker="^", c='c', s=100)
plt.scatter(x_est_x, x_est_y, marker="o", c='r', s=100)
plt.scatter(tx_pos_x, tx_pos_y, marker="*", c='k', s=100)
xmin = min([min(x_n_x), min(x_est_x), min(tx_pos_x)]) - 100
xmax = max([max(x_n_x), max(x_est_x), max(tx_pos_x)]) + 100
ymin = min([min(x_n_y), min(x_est_y), min(tx_pos_y)]) - 200
ymax = max([max(x_n_y), max(x_est_y), max(tx_pos_y)]) + 100
plt.axis([xmin, xmax, ymin, ymax])
plt.xlabel('X - Achse', fontsize=20)
plt.ylabel('Y - Achse', fontsize=20)
plt.grid()
plt.legend(['Wahre Position', 'Geschaetzte Position', 'Transmitter Antennen'], loc=3)  # best:0, or=1, ol=2, ul=3, ur=4
plt.title('Plot der wahren und geschaetzten Punkte', fontsize=25)
# plt.show()

'''Strecke im Linienplot'''
x_est_fehler = [None]*len((x_est_x))
for i in range(len(x_est_x)):
    x_est_fehler[i] = get_distance_1D(x_est_x[i], x_est_y[i])
# plt.figure(figsize=(12, 12))
plt.subplot(243)
plt.plot(range(1, (len(x_n)+1)), x_n_x)
plt.plot(x_est_x)
plt.plot(range(1, (len(x_n)+1)), x_n_y)
plt.plot(x_est_y)
plt.xlabel('Messungsnummer', fontsize=20)
plt.ylabel('Koordinate', fontsize=20)
plt.legend(['Wahre Position X-Koordinate', 'Geschaetzte X-Koordinate',
            'Wahre Position Y-Koordinate', 'Geschaetzte Y-Korrdinate'], loc=0)
plt.subplot(247)
x_est_fehler = [None]*len(x_est_x)
for i in range(3, len(x_n_x)):
    x_est_fehler[i] = get_distance_1D(x_est_x[i], x_n_x[i-1])
plt.plot(x_est_fehler)
for i in range(3, len(x_n_y)):
    x_est_fehler[i] = get_distance_1D(x_est_y[i], x_n_y[i-1])
plt.plot(x_est_fehler)
for i in range(3, len(x_est_list)):
    x_est_fehler[i] = get_distance_2D(x_est_list[i], x_n[i-1])
plt.plot(x_est_fehler)
x_est_fehler_ges_mean = [np.mean(x_est_fehler[3:])]*len(x_est_x)
plt.plot(x_est_fehler_ges_mean, '--')
plt.xlabel('Messungsnummer', fontsize=20)
plt.ylabel('Fehler', fontsize=20)
plt.legend(['Fehler X-Koordinate', 'Fehler Y-Koordinate', '(Gesamt-) Abstandsfehler', 'Mittlerer Gesamtfehler'], loc=0)
plt.subplot(248)
plt.hist(x_est_fehler[3:], 60, (0, 60))
plt.legend([ekf_param_Xtra])
plt.show()

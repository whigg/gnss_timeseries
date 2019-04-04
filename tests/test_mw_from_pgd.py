import matplotlib.pyplot as plt
import numpy as np
from random import randint
from gnss_timeseries.gnss_timeseries import GnssTimeSeries
from gnss_timeseries.stats import is_not_outlier
from gnss_timeseries.aux import are_close, are_close_arr

s_rate = 1.
t_step = 1/s_rate
n = 1000
t = np.linspace(0, (n-1)*t_step, n)
enu = [np.random.normal(0, 0.15, n) for _ in range(3)]
std_enu = [np.abs(np.random.normal(0.15, 0.02, n)) for _ in range(3)]

k_1quarter = int(0.25*n)
k_half = int(0.5*n)
k_3quarters = int(0.75*n)
k_5eigths = int(0.625*n)

m_reference = min(100, k_1quarter)
win_reference = int(round(t_step*m_reference))
m_gd = 600
win_gd = int(round(t_step*m_gd))
m_offset = 200
win_offset = int(round(t_step*m_offset))

print('{:.16f}'.format((t[1]-t[0])*s_rate))

k_event = randint(k_1quarter, k_half)
k_flag = randint(k_5eigths, k_3quarters)
t_event = t[k_event]
t_flag = t[k_flag]
t_5eights = t[k_5eigths]

t_aux = (t - t_event)*0.2
for x in enu:
    x[k_event:] += np.random.normal(0, 2)*(
            1 - np.exp(-0.2*t_aux[k_event:])*np.cos(t_aux[k_event:]))
    x += 2*np.exp(-0.01*(t-t_5eights)**2)*np.sin(2*t_aux)

k1 = k_event - m_reference
k2 = k_event + 1
enu_ref = []
var__enu_ref = []
for x, std_x in zip(enu, std_enu):
    x_aux = x[k1:k2]
    mask = is_not_outlier(x_aux, confidence=3, check_finite=True)
    enu_ref.append(np.mean(x_aux[mask]))
    aux = std_x[k1:k2][mask]
    aux *= aux
    n_ok = mask.sum()
    var__enu_ref.append(aux.sum()/(n_ok*n_ok))
t_segment = t[k1], t[k_event]

f, axes = plt.subplots(4, 1, sharex=True)

for x, x_ref, ax in zip(enu, enu_ref, axes):
    ax.plot(t, x, '-', linewidth=1.0)
    ax.plot(t_segment, (x_ref, x_ref), color='r',
            linewidth=1.0, linestyle='--')
    ax.axvline(t_event,  color='m', linewidth=1.0, linestyle='--')

k1 = k_event
k2 = min(n, k1+m_gd+1)
gd = np.zeros(k2-k1)
pgd = np.zeros(gd.size)
for x, x_ref in zip(enu, enu_ref):
    gd += (x[k1:k2] - x_ref)**2
gd = np.sqrt(gd)
gd_max = 0.
for k in range(gd.size):
    if gd[k] > gd_max:
        gd_max = gd[k]
    pgd[k] = gd_max

axes[3].plot(t[k1:k2], gd, '-', color='k', linewidth=1.0)
axes[3].plot(t[k1:k2], pgd, '--', color='r', linewidth=1.0)

# GnssTimeSeries instance
gt = GnssTimeSeries(length=t[-1], sampling_rate=s_rate,
                    window_offset=win_offset)
# add data
for k in range(n):
    gt.add_point({'coords': tuple(x[k] for x in enu),
                  'std_coords': tuple(x[k] for x in std_enu)}, t[k])

gt.eval_ref_values(t_event, window_ref=win_reference)
enu_ref_eval = gt.ref_values()[0]
print(enu_ref)
print(enu_ref_eval)
# TEST:  reference values  -->  check
for a, b in zip(enu_ref, enu_ref_eval):
    assert are_close(a, b, tol=1.e-14)
# TEST:  ground displacement time series  -->  check
gd_eval, t_gd_eval = gt.ground_displ_timeseries(window=win_gd)
assert are_close_arr(gd_eval, gd, tol=1.e-14)
print(np.abs(gd_eval-gd).max())
# TEST:  peak ground displacement time series  -->  check
pgd_eval, t_pgd_eval = gt.pgd_timeseries(window=win_gd)
assert are_close_arr(pgd_eval, pgd, tol=1.e-14)
plt.show()

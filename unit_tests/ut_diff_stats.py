import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timezone
from gnss_timeseries.stats import general_stats
f_stencil = '/home/francisco/Escritorio/GNSS_datos/Pisagua_2014_UTM/{:s}.UTM'
stations = ['MNMI', 'PB01', 'PB02', 'PB03', 'PB04',
            'PB05', 'PB06', 'PB08', 'PB11', 'PSGA']
ts_origin = datetime(2014, 4, 1, 23, 46, 45, tzinfo=timezone.utc).timestamp()
coord_names = ('x1', 'x2', 'x3')


def vel(t, x):
    return (x[1:] - x[:-1])/(t[1:] - t[:-1])


x1_list = []
x2_list = []
x3_list = []
ts_list = []
ts_aux = 0

for sta in stations:
    ts, x1, x2, x3 = np.loadtxt(f_stencil.format(sta), usecols=(3, 6, 7, 8),
                                unpack=True)
    n_origin = np.argmin(np.abs(ts - ts_origin))
    x1_list.append(x1[:n_origin]-x1[0])
    x2_list.append(x2[:n_origin]-x2[0])
    x3_list.append(x3[:n_origin]-x3[0])
    ts = ts[:n_origin]
    ts += ts_aux - ts[0]
    ts_aux = 3*ts[-1] - 2*ts[-2]
    ts_list.append(ts)

ts_ = np.hstack(ts_list)
x1_ = np.hstack(x1_list)
x2_ = np.hstack(x2_list)
x3_ = np.hstack(x3_list)
diff1_stats = general_stats(x1_[1:]-x1_[:-1], rel_threshold_outliers=3)
diff2_stats = general_stats(x2_[1:]-x2_[:-1], rel_threshold_outliers=3)
diff3_stats = general_stats(x3_[1:]-x3_[:-1], rel_threshold_outliers=3)
print('')
print('-'*75)
print('')
print('Quartiles: ',
      diff1_stats['quartiles'],
      diff2_stats['quartiles'],
      diff3_stats['quartiles'])
print('')
print('Thresholds: ',
      diff1_stats['threshold'],
      diff2_stats['threshold'],
      diff3_stats['threshold'])
print('')

fig_a, (ax_a1, ax_a2, ax_a3) = plt.subplots(3, 1, sharex='all')
ax_a1.plot(ts_, x1_, '.-r')
ax_a2.plot(ts_, x2_, '.-b')
ax_a3.plot(ts_, x3_, '.-k')

v1_ = vel(ts_ ,x1_)
v2_ = vel(ts_ ,x2_)
v3_ = vel(ts_ ,x3_)

tau = 0.5*(ts_[1:] + ts_[:-1])
fig_b, (ax_b1, ax_b2, ax_b3) = plt.subplots(3, 1, sharex='all')
ax_b1.plot(tau, v1_, '.-r')
ax_b2.plot(tau, v2_, '.-b')
ax_b3.plot(tau, v3_, '.-k')

is_out = np.abs(v1_) > diff1_stats['threshold']
aux = np.logical_and(is_out[1:], is_out[:-1])
print(aux.sum(), np.where(aux))
ax_b1.plot(tau[is_out], v1_[is_out], '.c')
is_out = np.abs(v2_) > diff2_stats['threshold']
aux = np.logical_and(is_out[1:], is_out[:-1])
print(aux.sum(), np.where(aux))
ax_b2.plot(tau[is_out], v2_[is_out], '.c')
is_out = np.abs(v3_) > diff3_stats['threshold']
aux = np.logical_and(is_out[1:], is_out[:-1])
print(aux.sum(), np.where(aux))
ax_b3.plot(tau[is_out], v3_[is_out], '.c')

plt.show()

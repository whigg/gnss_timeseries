import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timezone
from os.path import join, dirname, abspath
from timeseries.aux import are_close, are_all_close, are_close_vec
from gnss_timeseries.gnss_timeseries import GnssTimeSeries

# -------------------------------------------------------------
#       ****     TEST: GnssTimeSeries set_series      ****
# -------------------------------------------------------------

f_stencil = '/home/francisco/Escritorio/CSN/GNSS_datos/Pisagua_2014_UTM/{:s}.UTM'
# f_stencil = '/home/francisco/Escritorio/GNSS_datos/Pisagua_2014_UTM/{:s}.UTM'
stations = ['MNMI', 'PB01', 'PB02', 'PB03', 'PB04',
            'PB05', 'PB06', 'PB08', 'PB11', 'PSGA']
ts_origin = datetime(2014, 4, 1, 23, 46, 45, tzinfo=timezone.utc).timestamp()

index = 6
sta = stations[index]
ts, x1, x2, x3, std_x1, std_x2, std_x3 = np.loadtxt(
    f_stencil.format(sta), usecols=(3, 6, 7, 8, 9, 10, 11), unpack=True)
aux = ts[1:] - ts[:-1]
s_rate = np.nanmedian(aux)
print('Number gaps in data: {:d}'.format(np.sum(np.round(aux*s_rate) > 1)))
print('s_rate = {:.6e} Hz'.format(s_rate))

fig_a, axes = plt.subplots(6, 1, sharex='all',
                                            figsize=(16, 9))
axes[0].plot(ts, x1, '.-r')
axes[1].plot(ts, x2, '.-b')
axes[2].plot(ts, x3, '.-k')
plt.tight_layout()

# ====================================================== #
gnss_t_series = GnssTimeSeries(                          #
    length='2h', sampling_rate=s_rate, win_offset='1m',  #
    win_detect='10s', shifts_anomaly=('1s', '5s'))       #
# ====================================================== #

coords = x1, x2, x3, std_x1, std_x2, std_x3
gnss_t_series.set_series(coords, ts[-1])
coords_2, t = gnss_t_series.get(get_time=True)

n_data = ts.size
assert are_all_close(ts, t[-n_data:])
for f1, f2 in zip(coords, coords_2):
    assert np.all(np.isnan(f2[:-n_data]))
    assert are_all_close(f1, f2[-n_data:])


enu_diff, ts2 = gnss_t_series.get(layers='anomaly',
                                  get_time=True, as_dict=True)
print(enu_diff['dE_5s'][-n_data:-n_data+20])
print(enu_diff['dE_5s'][-n_data:-n_data+20])

axes[3].plot(ts2, enu_diff['dE_1s'], '--r')
axes[3].plot(ts2, enu_diff['dE_5s'], '--c')
axes[4].plot(ts2, enu_diff['dN_1s'], '--r')
axes[4].plot(ts2, enu_diff['dN_5s'], '--c')
axes[5].plot(ts2, enu_diff['dU_1s'], '--r')
axes[5].plot(ts2, enu_diff['dU_5s'], '--c')

n_diff = gnss_t_series._shifts_index_anomaly[0]
print(n_diff)
print(enu_diff['dE_1s'][-n_data:])
# print(enu_diff['dE_1s'][-n_data+1:] - (x1[1:] - x1[:-1]))
#

w = 5
aux = gnss_t_series.last((2*w+1)*gnss_t_series.t_step)
aux1 = gnss_t_series.get_around(t[-1] - w*gnss_t_series.t_step,
                                w*gnss_t_series.t_step)
aux2 = gnss_t_series.get_around(t[-1] - w*gnss_t_series.t_step,
                                w*gnss_t_series.t_step)
print('kjwqnfqwf')
print(x1[-2*w-1:])
print(aux[0])
print(aux1[0])
print(aux2[0])
# print(np.where(np.isnan(aux[0]))[0].size)
print(t[-1] + gnss_t_series.t_step)
plt.show()


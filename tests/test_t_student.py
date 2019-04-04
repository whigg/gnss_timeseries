import numpy as np
from time import clock
import matplotlib.pyplot as plt
from os.path import dirname, realpath, join
from gnss_timeseries.stats import t_student, critical_t, log_interpolator

x1 = np.array((30.02, 29.99, 30.11, 29.97, 30.01, 29.99))
x2 = np.array((29.89, 29.93, 29.72, 29.98, 30.02, 29.98))
t_equal, ddof_equal, t_cr_equal = t_student(
    x1, x2, equal_var=True, alpha='10%', t_critical=True)
t_not_equal, ddof_not_equal, t_cr_not_equal = t_student(
    x1, x2, equal_var=False, alpha='10%', t_critical=True)
print(t_equal, t_cr_equal, ddof_equal)
print(t_not_equal, t_cr_not_equal, ddof_not_equal)

path = join(realpath(dirname(dirname(__file__))),
            'gnss_timeseries', 't_student_table')
aux = np.loadtxt(path, skiprows=6, unpack=True)
# print(aux[0])
# print(aux[2])
print(critical_t('10%', ddof_equal))
print(critical_t('10%', ddof_not_equal))

# fig1 = plt.figure()
# ax1 = fig1.add_axes([0, 0, 1, 1])
# for a in aux[1:]:
#     ax1.plot(np.log(aux[0][2:]), np.log(a[2:]))
    # ax.plot(np.log(np.log(aux[0])), np.log(a))

fig2 = plt.figure()
ax2 = fig2.add_axes([0,0,1,1])

df = np.linspace(1, 500, 10000)
for a in aux[1:]:
    log_interp = log_interpolator(aux[0][:-1], a[:-1])
    ax2.loglog(df, log_interp(df), '-')
    ax2.loglog(aux[0], a, '.k')
plt.show()

# t0 = clock()
# for k in range(1000):
#     critical_t('10%', np.random.uniform(1, 500))
# print(clock()-t0)

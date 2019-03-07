import numpy as np
sec_min = 60  # 1 minute in seconds
sec_hour = 3600  # 1 hour in seconds
sec_day = 86400  # 1 day in seconds
sec_month = 2629800  # 1 month in seconds (30.4375 days)
sec_year = 31557600  # 1 year in seconds (365.25 days)
default_win_ref = 600
default_win_pgd = 300


def log_interpolator(x, y):
    ln_x = np.log(x)
    ln_y = np.log(y)

    def log_interp(x_eval):
        if isinstance(x_eval, np.ndarray):
            return np.array([log_interp(z) for z in x_eval])
        if x_eval < x[0]:
            return y[0]
        elif x_eval > x[-1]:
            return y[-1]

        index = -1
        for k in range(1, x.size):
            if x_eval < x[k]:
                index = k-1
                break
        theta = (np.log(x_eval) - ln_x[index])/(ln_x[index+1] - ln_x[index])
        return np.exp(ln_y[index]*(1-theta) + ln_y[index+1]*theta)

    return log_interp

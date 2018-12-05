"""Linear inverse problems and bayesian estimation with linear restriction."""
import numpy as np
from gnss_timeseries.stats import is_not_outlier
__docformat__ = 'reStructuredText en'


# -----------------------------------------------------------------------------
#                       Linear regression (one variable)
def linreg(x, y, stdev_y=None, weights=None,
           no_outliers=False, outliers_conf=4):
    r"""
    Linear regression

    Computes the coefficients :math:`a_0` and :math:`a_1` of the (weighted)
    least squares linear regression.

    :param x: :math:`x` values :math:`\{x_k\}_{k=1}^n`
    :param y: :math:`y` values :math:`\{y_k\}_{k=1}^n`.
    :param stdev_y: standard deviations of :math:`y`,
        :math:`\{\sigma_k\}_{k=1}^n (optional). This is equivalent to the
        statistical weights :math:`w_k = \frac{1}{{\sigma_k}^2}`.
    :param weights: statistical weights :math:`w_k = \frac{1}{{\sigma_k}^2}`.
        Only used if `std_y` is `None`.
    :param no_outliers: remove outliers?
    :param outliers_conf: confidence level: 1, 2, 3 or 4,
        which correspond to 90%, 95%, 99% and 99.9% two-tailed confidence
        respectively (normal distribution). Default: 4 (99.9%)
    :type x: numpy.ndarray
    :type y: numpy.ndarray
    :type stdev_y: numpy.ndarray or None
    :type weights: numpy.ndarray or None
    :type no_outliers: bool
    :type outliers_conf: int
    :return: :math:`a_1`, :math:`a_0`
    :rtype: (float, float)

    .. math:: \min_{a_0,\,a_1} \,\,
            \sum \limits_{k=1}^{n} \frac{(a_0 + a_1 x_k - y_k)^2}{{\sigma_k}^2}

    .. note:: this function is faster than *scipy.stats.linregress* by a factor
        between 1.07 up to 5.0 (larger for smaller sizes).
    """
    if no_outliers:
        return _linreg_no_outliers(x, y, stdev_y=stdev_y, weights=weights,
                                   confidence=outliers_conf)
    return _linreg_plain(x, y, stdev_y=stdev_y, weights=weights)


def _linreg_plain(x, y, stdev_y=None, weights=None):
    x_ave, y_ave, xy_ave, x2_ave = average((x, y, x*y, x*x),
                                           stdev=stdev_y, weights=weights)
    aux = x2_ave - x_ave*x_ave
    if abs(aux) < 1.e-300:
        return np.nan, np.nan
    else:
        a1 = (xy_ave - x_ave*y_ave)/aux
        return a1, y_ave - a1*x_ave


def _linreg_no_outliers(x, y, stdev_y=None, weights=None, confidence=4):
    # removing outliers
    wt = _weights(stdev_y, weights)
    mask_ok = is_not_outlier(y, confidence=confidence)
    c1 = np.nan
    c0 = np.nan
    for k in range(3):
        c1, c0, mask_ok_new = _linreg_no_outliers_step(
            x[mask_ok], y[mask_ok],
            None if wt is None else wt[mask_ok], confidence)
        n_new_ok = mask_ok_new.sum()
        if n_new_ok == mask_ok_new.size or n_new_ok < 0.75*mask_ok_new.size:
            break
        mask_ok[mask_ok] = mask_ok_new

    return c1, c0, mask_ok


def _linreg_no_outliers_step(x, y, weights, confidence):
    c1, c0 = _linreg_plain(x, y, stdev_y=None, weights=weights)
    mask_ok_new = is_not_outlier(c1 + c0*x - y, confidence=confidence)
    return c1, c0, mask_ok_new


# -----------------------------------------------------------------------------
#             Linear regression: pinning a point or fixing the slope
def linreg_pinned(xy_point, x, y, stdev_y=None, weights=None):
    delta_x = (x - xy_point[0])
    sx2, sxy = average((delta_x*delta_x, delta_x*(y - xy_point[0])),
                       stdev=stdev_y, weights=weights)
    return sx2/sxy


def linreg_fixed_slope(slope, x, y, stdev_y=None, weights=None):
    return average(y - slope*x, stdev=stdev_y, weights=weights)


def average(vec, stdev=None, weights=None):
    is_numpy_array = isinstance(vec, np.ndarray)
    w = _weights(stdev, weights)
    if w is None:
        if is_numpy_array:
            return vec.mean()
        return tuple(v.mean() for v in vec)
    if is_numpy_array:
        return w.dot(vec)
    return tuple(w.dot(v) for v in vec)


def _weights(stdev, weights):
    if stdev is None:
        if weights is None:
            return None
        w = weights
    else:
        w = 1./(stdev*stdev)
    return w/w.sum()


# -----------------------------------------------------------------------------
#                   Special Linear regression: localized
def linreg_local(x, y, x_center=None, stdev_y=None, weights=None,
                 damping_length=None, no_outliers=False, outliers_conf=4):
    if x_center is None:
        x_center = np.median(x)
    wt = _weights(stdev_y, weights)
    d_length = 0.1*x.ptp() if damping_length is None else damping_length
    aux = (x - x_center)/d_length
    aux = np.exp(-aux*aux)
    wt = aux if wt is None else wt*aux
    return linreg(x, y, stdev_y=None, weights=wt,
                  no_outliers=no_outliers,
                  outliers_conf=outliers_conf)

import numpy as np
from os.path import dirname, realpath, join
from .aux import log_interpolator
__docformat__ = 'reStructuredText en'
_alpha = (0.71933, 0.95292, 1.40946, 1.93927)
# alpha for p = 0.9, 0.95, 0.99 and 0.999 respectively.
# What is alpha? We assume a normal distribution. Let Q1, Q3 be the
# 25% and 75% quantiles respectively. Let z be the z-score and p the p-value.
# z(p) = Q3 + alpha*(Q3 - Q1) = (1 + 2*alpha)*Q3

_col_t_table = {'20%': 1, '10%': 2, '5%': 3, '2%': 4,
                '1%': 5, '.5%': 6, '.2%': 7, '.1%': 8}
path = join(realpath(dirname(dirname(__file__))),
            'gnss_timeseries', 't_student_table')
_t_table = np.loadtxt(path, skiprows=6, unpack=True)
_t_interp_tuple = tuple(log_interpolator(_t_table[0][:-1], _t_table[k][:-1])
                        for k in range(1, len(_t_table)))


def t_student(x1, x2, equal_var=True, alpha='1%', t_critical=False):
    r"""t-statistic for Student's t-test.

    :param x1: vector with first set of values
    :param x2: vector with second set of values
    :param equal_var: equal variances?
    :param alpha: two-tailed confidence parameter :math:`\alpha`. Valid values:
        '20%', '10%', '5%', '2%', '1%', '.5%', '.2%', '.1%'
    :param t_critical: return critical t-statistic value?
    :type x1: numpy.ndarray
    :type x2: numpy.ndarray
    :type equal_var: bool
    :type alpha: str
    :type t_critical: bool
    :return: t-statistic, degrees of freedom, (critical "t")
    """
    n1 = x1.size
    n2 = x2.size
    mean1 = x1.mean()
    mean2 = x2.mean()
    sum2_1 = np.sum((x1 - mean1)**2)
    sum2_2 = np.sum((x2 - mean2)**2)
    if equal_var:
        ddof = n1 + n2 - 2
        if n1 == n2:
            aux = (sum2_1 + sum2_2)/(n1*(n1-1))
        else:
            aux = (1./n1 + 1./n2)*(sum2_1 + sum2_2)/ddof
    else:
        aux1 = sum2_1/(n1*(n1-1))
        aux2 = sum2_2/(n2*(n2-1))
        aux = aux1 + aux2
        ddof = aux*aux/(aux1*aux1/(n1-1) + aux2*aux2/(n2-1))
    t = (mean1 - mean2)/np.sqrt(aux)

    if t_critical:
        return t, ddof, critical_t(alpha, ddof)
    return t, ddof


def var(x):
    return np.var(x, ddof=1)


def critical_t(alpha, ddof):
    r"""Critical value of t-statistic to reach a level of confidence.

    :param alpha: two-tailed confidence parameter :math:`\alpha`. Valid values:
        '20%', '10%', '5%', '2%', '1%', '.5%', '.2%', '.1%'
    :param ddof: degrees of freedom
    :type alpha: str
    :type ddof: int or float
    :return:
    """
    index_alpha = _col_t_table[alpha]
    if ddof > _t_table[0][-2]:
        return _t_table[index_alpha][-1]
    if isinstance(ddof, int) and ddof < 41:
        return _t_table[index_alpha][ddof - 1]
    return _t_interp_tuple[index_alpha-1](ddof)


# -----------------------------------------------------------------------------
#                       Outliers
def is_outlier(x, check_finite=False, confidence=3):
    """Boolean mask with outliers

    :param x: vector
    :param check_finite:
    :param confidence: confidence level: 1, 2, 3 or 4, which correspond to
        90%, 95%, 99% and 99.9% two-tailed confidence respectively (normal
        distribution). Default: 3 (99%)
    :type x: numpy.ndarray
    :type check_finite: bool
    :type confidence: int
    :return: vector with condition "is `x` outlier?"
    """
    return np.logical_not(
        is_not_outlier(x, check_finite=check_finite, confidence=confidence))


def is_not_outlier(x, confidence=3, check_finite=False):
    """Boolean mask with non-outliers

    :param x: vector
    :param check_finite: check for NaN or Inf values?
    :param confidence: confidence level: 1, 2, 3 or 4, which correspond to
        90%, 95%, 99% and 99.9% two-tailed confidence respectively (normal
        distribution). Default: 3 (99%)
    :type x: numpy.ndarray
    :type check_finite: bool
    :type confidence: int
    :return: vector with condition "is `x` outlier?"
    """
    if check_finite:
        q = np.nanquantile(x, (0.25, 0.75))
    else:
        q = np.quantile(x, (0.25, 0.75))
    delta = _alpha[confidence-1]*(q[1] - q[0])  # Q3 - Q1: interquartile range
    return np.logical_and(q[0] - delta < x, x < q[1] + delta)


def eval_no_outliers(func, x, confidence=3, check_finite=False,
                     get_mask_ok=False):
    """Evaluates a function of a vector removing its outliers first.

    :param x:  vector
    :param func: function
    :param confidence: confidence level: 1, 2, 3 or 4, which correspond to
        90%, 95%, 99% and 99.9% two-tailed confidence respectively (normal
        distribution). Default: 3 (99%)
    :param check_finite: check for NaN or Inf values?
    :param get_mask_ok: return boolean mask of not-outliers?
    :return: evaluation of the function at the vector without outliers,
        (is-outlier mask).
    """
    mask = is_not_outlier(x, confidence=confidence, check_finite=check_finite)
    if get_mask_ok:
        return func(x[mask]), mask
    return func(x[mask])


# -----------------------------------------------------------------------------
#                       General statistics
def general_stats(x, rel_threshold_outliers=2.5):
    r"""Computes some general statistical properties of a set of values
    :math:`\mathbf{X}=\{x_j\}_{j=1}^n`

    * quartiles :math:`Q_1(\mathbf{X}),\,Q_2(\mathbf{X}),\,Q_3(\mathbf{X})`
    * inter-quartile range
        :math:`IQR(\mathbf{X}) = Q_3(\mathbf{X}) - Q_1(\mathbf{X})`
    * outlier threshold :math:`\theta = \beta \cdot IQR(\mathbf{X})`
    * mean: :math:`\mu_\mathbf{x'} = ave(\mathbf{X'})`
    * standard deviation. :math:`\sigma_\mathbf{x'} = std(\mathbf{X'})`
    * mean/IQR: :math:`\mu_\mathbf{x'}/IQR(\mathbf{X})`

    where :math:`\mathbf{X}'=\{x \in \mathbf{X} \,:\,\, |x-m_x| < \theta\}`
    is the set of points without its outliers.

    :param x: vector of values
    :param rel_threshold_outliers: outlier threshold :math:`\beta` relative to
        the inter-quartile range.
    :return: dictionary with general statistics
    """
    quartiles = np.nanquantile(x, (0.25, 0.5, 0.75))
    iqr = quartiles[2] - quartiles[0]
    threshold = rel_threshold_outliers*iqr
    is_ok = np.abs(x - quartiles[1]) < threshold
    mean = x[is_ok].mean()
    gen_stats = {'quartiles': quartiles,
                 'threshold': threshold,
                 'mean': mean,
                 'std': x[is_ok].std(),
                 'mean/iqr': mean/iqr}
    return gen_stats

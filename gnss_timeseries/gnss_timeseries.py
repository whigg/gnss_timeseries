import numpy as np
from timeseries.timeseries import LayeredTimeSeries
from .aux import sec_min, sec_hour, sec_day, sec_month, sec_year
from .stats import eval_no_outliers

_dict_time_in_sec = {'s': 1, 'm': sec_min, 'h': sec_hour, 'D': sec_day,
                     'M': sec_month, 'Y': sec_year}
_coords_layer_id = 'coords'
_all_labels = ('E', 'N', 'U', 'stdE', 'stdN', 'stdU')
_coord_labels = ('E', 'N', 'U')
_std_coord_labels = ('stdE', 'stdN', 'stdU')
coord_layers = ('coords', 'std_coords')
_dict_nan_pgd = dict(PGD=np.nan, t_PGD=np.nan)


def parse_time(t_str):
    if isinstance(t_str, str):
        return float(t_str[:-1])*_dict_time_in_sec[t_str[-1]]
    return t_str


def parse_frequency(t_freq):
    if isinstance(t_freq, str):
        return float(t_freq[:-2])/_dict_time_in_sec[t_freq[-1]]
    return t_freq


class GnssTimeSeries(LayeredTimeSeries):
    """
    This class implements a ring-buffer to store GNSS coordinates of a station,
    in an evenly spaced time series. It is ideal to work with real time data.

    .. note::

        The majority of the API is inherited from
        the parent class :py:class:`timeseries.timeseries.LayeredTimeSeries`.
        It is strongly recommended to read the docstrings of this class and its
        parent class :py:class:`timeseries.timeseries.TimeSeries`

    New data points are appended at the end of the buffer, adding empty points
    before if necessary. Older data points can also be added to fill empty
    intervals.
    """

    def __init__(self, length='1h', sampling_rate='1/s',
                 half_window_offset='10m'):
        """

        :param length: length of the timeseries. It can be a float/int value
            in seconds or a string like '10m', '1h', '12h'.
        :param sampling_rate: sampling rate. It can be a float/int value
            in hertz or a string like '1/s', '3/s', '10/m'.
        :type length: str or float
        :type sampling_rate: str
        """
        layers_dict = {'coords': _coord_labels,
                       'std_coords': _std_coord_labels}
        super(GnssTimeSeries, self).__init__(
            parse_time(length), parse_frequency(sampling_rate),
            layers_dict, 'coords')
        # window to compute the mean of a coordinate before and after a
        # coseismic offset
        self.half_win_offset = parse_time(half_window_offset)

    def set_window_offset(self, window_offset):
        self.half_win_offset = parse_time(window_offset)

    def get_around(self, t, window, layers=coord_layers,
                   get_time=False, as_dict=False):
        """Gets one or more layers in a window around a point. By default, it
        gets the ENU coordinates.

        :param t: unix time
        :param window:
        :param layers:
        :param get_time:
        :param as_dict:
        :return:
        """
        return self.interval(max(t-window, self._t_first),
                             min(t + window, self.t_last), layers=layers,
                             get_time=get_time, as_dict=as_dict)

    def eval_offset(self, t_eval, conf_outliers=3, check_finite=True):
        """Computes the Static Offset given a reference time

        :param t_eval: reference unix time
        :param conf_outliers: degree of confidence to remove outliers
        :param check_finite: check for NaN or inf values
        :return: dictionary with offsets, their standard deviations and
            the means of the coordinates before and after the event.
        """
        offset_dict = dict(t_offset=t_eval, half_win=self.half_win_offset)
        if np.isnan(t_eval) or np.isnan(self.t_last):
            for k in range(3):
                c = _coord_labels[k]
                offset_dict[c] = np.nan
                offset_dict['std_' + c] = np.nan
                offset_dict['pre_mean_' + c] = np.nan
                offset_dict['post_mean_' + c] = np.nan
            return offset_dict
        all_fields, t = self.get_around(
            t_eval, self.half_win_offset, layers=('coords', 'std_coords'),
            get_time=True)
        n = int(round(0.5*(all_fields['coords'][0].size-1)))
        enu = all_fields['coords']
        std_enu = all_fields['coords']
        for k in range(3):
            c = _coord_labels[k]
            mean_pre, mask_ok_pre = eval_no_outliers(
                np.mean, enu[k][:n], confidence=conf_outliers,
                check_finite=check_finite, get_mask_ok=True)
            mean_post, mask_ok_post = eval_no_outliers(
                np.mean, enu[k][n:], confidence=conf_outliers,
                check_finite=check_finite, get_mask_ok=True)
            offset_dict[c] = mean_post - mean_pre
            c = '_' + c
            if mask_ok_post.sum() == 0 or mask_ok_pre.sum() == 0:
                offset_dict['std' + c] = np.nan
                offset_dict['pre_mean' + c] = np.nan
                offset_dict['post_mean' + c] = np.nan
                continue
            aux = std_enu[k][:n][mask_ok_pre]
            aux *= aux
            var1 = aux.sum()
            var1 /= mask_ok_pre.sum()**2
            aux = std_enu[k][n:][mask_ok_post]
            aux *= aux
            var2 = aux.sum()
            var2 /= mask_ok_post.sum()**2

            offset_dict['std' + c] = 0.5*np.sqrt(var1 + var2)
            offset_dict['pre_mean' + c] = mean_pre
            offset_dict['post_mean' + c] = mean_post
        return offset_dict

    def eval_pgd(self, t_origin, t_s=None, t_tol=120,
                 only_hor=True):
        """Computes the Peak Ground Displacement (PGD) and the time of its
        occurrence.

        :param t_origin: origin time of earthquake.
        :param t_s: reference time of the arrival of S-waves
        :param t_tol: maximum delay between t_s and the PGD.
        :param only_hor: only horizontal coordinates?
        :return: PGD, time of occurrence of PGD.
        """
        if np.isnan(self.t_last):
            return _dict_nan_pgd
        t1 = t_origin - 180
        if t_s is None:
            t_s = t_origin
            t_tol = 600
        t2 = t_s + t_tol

        enu, t = self.interval(t1, t2, get_time=True, check_input=True)
        if t.size == 0:
            return _dict_nan_pgd
        index_eval = index_eval_factory(t)
        if t_origin is None:
            i_origin = int(0.25*t.size)
        else:
            i_origin = index_eval(t_origin)
            if i_origin < 0 or i_origin >= t.size:
                return _dict_nan_pgd
        # assumption: if "E" is NaN "N" and "U" also are.
        mask = np.logical_not(np.isnan(enu[0]))
        i_s = index_eval(t_s)
        n_ref = mask[:i_origin].sum()
        n_signal = mask[i_s:].sum()
        if n_ref < 6 or n_signal < 4:
            return _dict_nan_pgd
        aux = np.zeros(t.size - i_s)
        displ_2 = np.zeros_like(aux)
        for k in range(2 if only_hor else 3):
            aux[:] = enu[k][i_s:]
            aux -= np.nanmedian(enu[k][:i_origin])
            displ_2 += aux*aux
        k_max = np.nanargmax(displ_2)
        return dict(PGD=np.sqrt(displ_2[k_max]), t_PGD=t[i_s:][k_max])

    def pgd_timeseries(self, t_origin, tau=10, window=600):
        gd, t = self.ground_displ_timeseries(
            t_origin, tau=tau, window=window)
        if gd is None:
            return None, None
        if np.isfinite(gd).sum() < 0.75*gd.size:
            return None, None
        return np.maximum.accumulate(np.nan_to_num(gd)), t

    def ground_displ_timeseries(self, t_origin, tau=10, window=600):
        if np.isnan(self.t_last):
            return None, None
        k_tau = int(round(tau*self.s_rate))
        coords, t = self.interval(
            t_origin - tau, min(self.t_last, t_origin+window), get_time=True)
        aux = np.zeros(coords[0].size)
        for a in range(3):
            x = coords[a] - coords[a][:k_tau].mean()
            aux += x*x
        return np.sqrt(aux), t

    def detect_wave(self, t_guess=None, **kwargs):
        """
        Detect seismic wave.

        :param t_guess: guess of timestamp at the first arrival
        :param kwargs: key-worded arguments
        :return:
        """
        pass


def index_eval_factory(t):
    aux = (t.size - 1)/(t[-1] - t[0])

    def index_eval(tau):
        return int(round(aux*(tau - t[0])))

    return index_eval

import numpy as np
from timeseries.timeseries import LayeredTimeSeries
from .aux import (sec_min, sec_hour, sec_day, sec_month, sec_year,
                  default_win_ref, default_win_pgd)
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

    def __init__(self, length='1h', sampling_rate='1/s', window_offset='10m'):
        """

        :param length: length of the timeseries. It can be a float/int value
            in seconds or a string like '10m', '1h', '12h'.
        :param sampling_rate: sampling rate. It can be a float/int value
            in hertz or a string like '1/s', '3/s', '10/m'.
        :param window_offset: length of window where the offset is computed
        :type length: str or float
        :type sampling_rate: str or float
        :type window_offset: str or float
        """
        layers_dict = {'coords': _coord_labels,
                       'std_coords': _std_coord_labels}
        super(GnssTimeSeries, self).__init__(
            parse_time(length), parse_frequency(sampling_rate),
            layers_dict, 'coords')
        # window to compute the mean of a coordinate before and after a
        # coseismic offset
        self.win_offset = parse_time(window_offset)
        self._ref_values_are_set = False
        self._enu_ref = {c: np.nan for c in _coord_labels}
        self._var_enu_ref = {c: np.nan for c in _coord_labels}
        self._t_origin = -1
        self._win_ref_values = -1

    def set_window_offset(self, window_offset):
        self.win_offset = parse_time(window_offset)

    def ref_values(self):
        return (tuple(self._enu_ref[c] for c in _coord_labels),
                self._win_ref_values, self._t_origin)

    def ref_values_are_set(self):
        return self._ref_values_are_set

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

    def _eval_mean_values(self, t_start, t_end, min_data_points=4,
                          conf_outliers=3, check_finite=True):
        """Computes mean and variance of coordinates in a window
        before an event. Removes outliers.

        :param t_start: start of the window
        :param t_end: end of the window
        :param min_data_points: minimum valid data points (non-outliers) to
            compute a value.
        :param conf_outliers: confidence level: 1, 2, 3 or 4.
        :param check_finite: check for NaN or Inf values?
        :return: dictionary with the means, dictionary with the variances
        """
        values = self.interval(t_start, t_end,
                               get_time=False, check_input=True,
                               layers=coord_layers)
        enu = values['coords']
        std_enu = values['coords']
        enu_mean = dict()
        var_enu_mean = dict()
        for k in range(3):
            c = _coord_labels[k]
            mean_x, mask_ok = eval_no_outliers(
                np.mean, enu[k], confidence=conf_outliers,
                check_finite=check_finite, get_mask_ok=True)
            n_ok = mask_ok.sum()
            if n_ok < min_data_points:
                enu_mean[c] = np.nan
                var_enu_mean[c] = np.nan
            else:
                enu_mean[c] = mean_x
                aux = std_enu[k][mask_ok]
                aux *= aux
                var_enu_mean[c] = aux.sum()/(n_ok*n_ok)
        return enu_mean, var_enu_mean

    def eval_ref_values(self, t_origin, window_ref=default_win_ref,
                        force_eval_ref_values=False, **kwargs_mean):
        """Computes reference values of coordinates as a mean in a window
        before an event. Removes outliers.

        :param t_origin: approximate origin time
        :param window_ref: length of the time window
        :param force_eval_ref_values: force evaluation of reference values?
        :param kwargs_mean: key-worded arguments. See the method
            :py:func:`_eval_mean_values`
        """
        if (self.t_last_not_set() or
                (self._ref_values_are_set and not force_eval_ref_values)):
            return False
        enu_mean, var_enu_mean = self._eval_mean_values(
            t_origin - window_ref, t_origin, **kwargs_mean)
        self._enu_ref.update(enu_mean)
        self._var_enu_ref.update(var_enu_mean)
        self._t_origin = t_origin
        self._win_ref_values = window_ref
        # all 3 reference values must be non-NaN
        self._ref_values_are_set = sum(map(np.isnan, enu_mean.values())) == 0
        return True

    def _clear_ref_values(self):
        self._enu_ref.clear()
        self._var_enu_ref.clear()
        self._t_origin = -1
        self._win_ref_values = -1
        self._ref_values_are_set = False

    def eval_offset(self, t_eval, t_origin=None, window_ref=default_win_ref,
                    force_eval_ref_values=False, **kwargs_mean):
        """Computes the Static Offset given a reference time

        :param t_eval: reference unix time
        :param t_origin: origin time. Only used if reference values are
            computed.
        :param window_ref: window for evaluation of reference values
        :param force_eval_ref_values: force evaluation of reference values?
        :return: dictionary with offsets, their standard deviations and
            the means of the coordinates before and after the event.

        .. warning::
            The reference values must be computed beforehand
        """
        offset_dict = dict(t_origin=self._t_origin,
                           win_ref_val=self._win_ref_values,
                           t_offset=t_eval,
                           win_offset=self.win_offset)
        if np.isnan(t_eval) or self.t_last_not_set():
            for k in range(3):
                c = _coord_labels[k]
                offset_dict[c] = np.nan
                offset_dict['std_' + c] = np.nan
                offset_dict['ref_val_' + c] = np.nan
                offset_dict['post_mean_' + c] = np.nan
            return offset_dict

        self.eval_ref_values(
            t_origin, window_ref=window_ref,
            force_eval_ref_values=force_eval_ref_values, **kwargs_mean)


        enu_mean, var_enu_mean = self._eval_mean_values(
            t_eval, t_eval + self.win_offset, **kwargs_mean)

        for k in range(3):
            c = _coord_labels[k]
            if np.isnan(enu_mean[c]):
                offset_dict[c] = np.nan
                offset_dict['std_' + c] = np.nan
                offset_dict['ref_val_' + c] = np.nan
                offset_dict['post_mean_' + c] = np.nan
                continue
            offset_dict[c] = enu_mean[c] - self._enu_ref[c]
            offset_dict['std_' + c] = 0.5*np.sqrt(self._var_enu_ref[c] +
                                                  var_enu_mean[c])
            offset_dict['ref_val_' + c] = self._enu_ref[c]
            offset_dict['post_mean_' + c] = enu_mean[c]
        return offset_dict

    def eval_pgd(self, t_origin, t_s=None, window_pgd=default_win_pgd,
                 only_hor=True, window_ref=default_win_ref,
                 force_eval_ref_values=False, **kwargs_mean):
        """Computes the Peak Ground Displacement (PGD) and the time of its
        occurrence.

        :param t_origin: origin time of earthquake.
        :param t_s: reference time of the arrival of S-waves
        :param window_pgd: maximum delay between t_s and the PGD.
        :param only_hor: only horizontal coordinates?
        :param window_ref: window for evaluation of reference values
        :param force_eval_ref_values: force evaluation of reference values?
        :return: PGD, time of occurrence of PGD.
        """
        if not self.eval_ref_values(
                t_origin, window_ref=window_ref,
                force_eval_ref_values=force_eval_ref_values, **kwargs_mean):
            return _dict_nan_pgd
        if t_s is None:
            t_s = t_origin
        t_end = t_s + window_pgd

        enu, t = self.interval(t_s, t_end,
                               get_time=True, check_input=True)
        if t.size < kwargs_mean.get('min_data_points', 4):
            return _dict_nan_pgd
        # assumption: if "E" is NaN "N" and "U" also are.
        mask = np.logical_not(np.isnan(enu[0]))
        n_ok = mask.sum()
        if n_ok < 4:
            return _dict_nan_pgd
        aux = np.zeros(t.size)
        displ_2 = np.zeros_like(aux)
        for k in range(2 if only_hor else 3):
            aux[:] = enu[k]
            aux -= self._enu_ref[_coord_labels[k]]
            displ_2 += aux*aux
        try:
            k_max = np.nanargmax(displ_2)
            return dict(PGD=np.sqrt(displ_2[k_max]), t_PGD=t[k_max])
        except ValueError:
            return _dict_nan_pgd

    def pgd_timeseries(self, t_origin=None, window=600, **kwargs_ref_values):
        gd, t = self.ground_displ_timeseries(t_origin=t_origin, window=window,
                                             **kwargs_ref_values)
        if gd is None:
            return None, None
        if np.isfinite(gd).sum() < 0.75*gd.size:
            return None, None
        return np.maximum.accumulate(np.nan_to_num(gd)), t

    def ground_displ_timeseries(self, t_origin=None, window=600,
                                **kwargs_ref_values):
        if self.t_last_not_set():
            return None, None
        self.eval_ref_values(t_origin, **kwargs_ref_values)
        if t_origin is None:
            t_origin = self._t_origin
        coords, t = self.interval(
            t_origin, min(self.t_last, t_origin+window), get_time=True)
        aux = np.zeros(coords[0].size)
        for a in range(3):
            x = coords[a] - self._enu_ref[_coord_labels[a]]
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

    def clear(self):
        self._clear_ref_values()
        super().clear()


def index_eval_factory(t):
    aux = (t.size - 1)/(t[-1] - t[0])

    def index_eval(tau):
        return int(round(aux*(tau - t[0])))

    return index_eval

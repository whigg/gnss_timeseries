import numpy as np
from gnss_timeseries.gnss_timeseries import (GnssTimeSeries, parse_time,
                                             parse_frequency)


class NetworkTimeSeries:
    """
    This class stores data of GNSS stations using GnssTimeSeriesSimple
    instances

    :ivar _station_ts: list of instances of
        :py:class:`gnss_timeseries.gnss_timeseries.GnssTimeSeriesSimple`
    """

    def __init__(self, length='1h', sampling_rate='1/s',
                 half_window_offset='10m', **kwargs_other):
        self.n_sta = 0
        self._station_ts = []
        self._ref_coords = []
        self._codes = []
        self._code2index = dict()
        self._names = []
        self.half_window_offset = parse_time(half_window_offset)
        self.ts_length = parse_time(length)
        self.s_rate = parse_frequency(sampling_rate)
        self._kwargs_other = kwargs_other

    def half_win_offset(self):
        return self.half_window_offset

    def set_window_offset(self, half_window_offset):
        self.half_window_offset = half_window_offset
        for ts in self._station_ts:
            ts.set_window_offset(half_window_offset)

    def station_timeseries(self, sta):
        """Buffer of the a station (GnssTimeSeriesSimple)

        :param sta: station index or code
        :return: buffer
        """
        return self._station_ts[self._sta2index(sta)]

    def add_station(self, code, ref_coords=None, name=''):
        """New station in the buffer

        :param code: station's 4 letter code
        :param ref_coords: (longitude, latitude)
        :param name: station's name
        """
        self._code2index[code] = self.n_sta
        self.n_sta += 1
        self._codes.append(code)
        self._names.append(name)
        self._ref_coords.append(ref_coords)
        self._station_ts.append(GnssTimeSeries(
            length=self.ts_length, sampling_rate=self.s_rate,
            half_window_offset=self.half_window_offset))

    def station_is_available(self, sta_code):
        return sta_code in self._codes

    def get_coords(self, sta, get_time=True, as_dict=False):
        return self.station_timeseries(sta).get(
            get_time=get_time, as_dict=as_dict)

    def get_coords_near(self, sta, t, layers=None, as_dict=False):
        return self.station_timeseries(sta).get_point(
            t, layers=layers, as_dict=as_dict)

    def get_interval(self, sta, t_begin, t_end, layers=None,
                     as_dict=False, get_time=True):
        return self.station_timeseries(sta).interval(
            t_begin, t_end, layers=layers, as_dict=as_dict, get_time=get_time)

    def get_last(self, sta, t_window, layers=None,
                 as_dict=False, get_time=True):
        return self.station_timeseries(sta).last(
            t_window, layers=layers, as_dict=as_dict, get_time=get_time)

    def get_first(self, sta, t_window, layers=None,
                  as_dict=False, get_time=True):
        return self.station_timeseries(sta).first(
            t_window, layers=layers, as_dict=as_dict, get_time=get_time)

    def get_layers(self, sta, layers, get_time=True, as_dict=False):
        return self.station_timeseries(sta).get(
            layers=layers, get_time=get_time, as_dict=as_dict)

    def add_point_to_station(self, sta, coords, std_coords, t, **kwargs):
        self.station_timeseries(sta).add_point(
            dict(coords=coords, std_coords=std_coords), t, **kwargs)

    def set_station_timeseries(self, sta, coords, std_coords, t,
                               check_sampling_rate=False):
        sta_timeseries = self.station_timeseries(sta)
        if check_sampling_rate:
            beta = np.median(t[1:] - t[:-1])*sta_timeseries.s_rate
            if abs(beta - 1.0) < 1.e-8:
                coords_aux = coords
                std_coords_aux = std_coords
            else:
                n = t.size
                m = int(round(beta))
                n_aux = (n-1)*m + 1
                coords_aux = []
                std_coords_aux = []
                for k in range(3):
                    x = np.full(n_aux, np.nan)
                    x[::m] = coords[k]
                    coords_aux.append(x)
                    s = np.full(n_aux, np.nan)
                    s[::m] = std_coords[k]
                    std_coords_aux.append(s)
        else:
            coords_aux = coords
            std_coords_aux = std_coords

        self.station_timeseries(sta).set_series(
            {'coords': coords_aux, 'std_coords': std_coords_aux}, t[-1])

    def ref_coords(self, code):
        """Reference coordinates of a station

        :param code: station index or code
        :return: longitude, latitude
        """
        return self._ref_coords[self._sta2index(code)]

    def station_codes(self):
        return self._codes

    def station_code(self, index):
        return self._codes[index]

    def station_name(self, code):
        return self._names[self._sta2index(code)]

    def station_index(self, code):
        return self._code2index[code]

    def _sta2index(self, code):
        if isinstance(code, int):
            return code
        return self._code2index[code]

    def eval_pgd(self, t_interval_dict=None, only_hor=True, t_ref_dict=None):
        pgd_dict = dict()
        if t_ref_dict is None:
            t_ref_dict = dict()
        if t_interval_dict is None:
            t_interval_dict = dict()
        for code, index in self._code2index.items():
            pgd_dict[code] = self.eval_pgd_at_station(
                code, t_interval=t_interval_dict.get(code), only_hor=only_hor,
                t_ref=t_ref_dict.get(code))
        return pgd_dict

    def eval_offset(self, t_eval_dict, conf_outliers=3, check_finite=True):
        offset_dict = dict()
        for code, index in self._code2index.items():
            offset_dict[code] = self.eval_offset_at_station(
                code, t_eval_dict.get(code),
                conf_outliers=conf_outliers, check_finite=check_finite)
        return offset_dict

    def eval_pgd_at_station(self, code, t_interval=None, only_hor=True,
                            t_ref=None):
        return self._station_ts[self._code2index[code]].eval_pgd(
            t_interval=t_interval, only_hor=only_hor, t_ref=t_ref)

    def eval_offset_at_station(self, code, t_eval,
                               conf_outliers=3, check_finite=True):
        return self._station_ts[self._code2index[code]].eval_offset(
            t_eval=t_eval, conf_outliers=conf_outliers,
            check_finite=check_finite)

    def set_win_offset(self, win_offset):
        win = parse_time(win_offset)
        for ts in self._station_ts:
            ts.win_offset = win

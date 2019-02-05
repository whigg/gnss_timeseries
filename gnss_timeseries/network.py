from math import sqrt, isfinite
import numpy as np
from gnss_timeseries.gnss_timeseries import (GnssTimeSeries, parse_time,
                                             parse_frequency)
from geoproj.proj import TransverseMercator


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
        self._tm = TransverseMercator(0., 0.)

    def half_win_offset(self):
        return self.half_window_offset

    def available_window(self):
        if self.n_sta == 0:
            return (np.nan, np.nan), np.nan
        t_min = np.inf
        t_max = -np.inf
        t_oldest = np.inf
        for index in range(self.n_sta):
            ts = self._station_ts[index]
            aux = ts.time_range()
            if aux[0] < t_min:
                t_min = aux[0]
            if aux[1] > t_max:
                t_max = aux[1]
            if ts.t_oldest < t_oldest:
                t_oldest = ts.t_oldest
        return (t_min, t_max), t_oldest

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

    def clear_data_at(self, sta):
        """Clears the data at one station

        :param sta: station index or code
        """
        self.station_timeseries(sta).clear()

    def clear_all_data(self):
        for ts in self._station_ts:
            ts.clear()

    def station_buffer_is_empty(self, sta_code):
        return np.isnan(self.station_timeseries(sta_code).t_last)

    def station_is_available(self, sta_code):
        return sta_code in self._codes

    def get_coords(self, sta, get_time=True, layers=None, as_dict=False):
        return self.station_timeseries(sta).get(
            layers=layers, as_dict=as_dict, get_time=get_time)

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

    def eval_pgd(self, t_interval_dict=None, only_hor=False, t_ref_dict=None):
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

    def eval_pgd_at_station(self, code, t_interval=None, only_hor=False,
                            t_ref=None):
        return self._station_ts[self._code2index[code]].eval_pgd(
            t_interval=t_interval, only_hor=only_hor, t_ref=t_ref)

    def eval_offset_at_station(self, code, t_eval,
                               conf_outliers=3, check_finite=True):
        return self._station_ts[self._code2index[code]].eval_offset(
            t_eval=t_eval, conf_outliers=conf_outliers,
            check_finite=check_finite)

    def eval_pgd_and_mw(self, hipocenter_coords, t_interval_dict=None,
                        only_hor=False, t_ref_dict=None):
        aux = self.eval_pgd(t_interval_dict=t_interval_dict,
                            only_hor=only_hor, t_ref_dict=t_ref_dict)
        return self.mw_from_pgd(
            hipocenter_coords,
            {code: value['PGD'] for code, value in aux.items()})

    def mw_from_pgd(self, hipocenter_coords, pgd_dict, max_distance=800):
        distance_dict = self._distance_dict(hipocenter_coords,
                                            max_distance=max_distance)
        results_dict = dict()
        for code, r in distance_dict.items():
            try:
                pgd = pgd_dict[code]
                if isfinite(pgd):
                    results_dict[code] = (mw_melgar(100*pgd, r), r)
            except KeyError:
                pass
        return results_dict

    def _distance_dict(self, hipocenter_coords, max_distance=800):
        self._tm.reset(hipocenter_coords[0], hipocenter_coords[1])
        depth_2 = hipocenter_coords[2]*hipocenter_coords[2]
        distance_dict = dict()
        for code, ref_coords in zip(self._codes, self._ref_coords):
            if ref_coords is None:
                continue
            x, y = self._tm(*ref_coords)
            r = sqrt(depth_2 + (x*x + y*y)*1.e-6)
            if r < max_distance:
                distance_dict[code] = r
        return distance_dict

    def mw_timeseries_from_pgd(self, hipocenter_coords, t_origin, vel_mask=3.,
                               sta_list=None, tau=10, max_distance=800,
                               window=300):
        pgd_dict = self.pgd_timeseries(t_origin, sta_list=sta_list, tau=tau,
                                       window=window)

        distance_dict = self._distance_dict(hipocenter_coords,
                                            max_distance=max_distance)
        # times at which each station is reached by the mask

        t_mask = []
        codes = []
        for code, r in distance_dict.items():
            if pgd_dict[code][0] is None:  # not enough data
                continue
            t_mask.append(t_origin + r/vel_mask)
            codes.append(code)
        if len(codes) == 0:
            return None, None
        t_mask = np.array(t_mask)
        # minimum and maximum times
        t_min = np.inf
        t_max = -np.inf
        for code in codes:
            t = pgd_dict[code][1]
            aux = t[0]
            if aux < t_min:
                t_min = aux
            aux = t[-1]
            if aux > t_max:
                t_max = aux
        t_step = 1./self.s_rate
        t_mw = np.arange(t_min-t_origin, t_max-t_origin+0.5*t_step, t_step)
        mw = np.zeros(t_mw.size)
        count = np.zeros(t_mw.size, dtype=int)

        for code, t_m in zip(codes, t_mask):
            pgd, t = pgd_dict[code]
            if t_m > t[-1]:
                continue
            k = np.argmin(np.abs(t-t_m))
            aux = mw_melgar(100*pgd[k:], distance_dict[code])
            i1 = np.argmin(np.abs(t_mw - (t[k] - t_origin)))
            i2 = i1 + aux.size
            mw[i1:i2] += aux
            count[i1:i2] += 1
        count[count == 0] = 1
        mw /= count
        mw[mw < 1.e-5] = np.nan
        return mw, t_mw

    def pgd_timeseries(self, t_origin, sta_list=None,
                       tau=10, window=300):
        if sta_list is None:
            sta_list = self.station_codes()
        pgd_dict = dict()
        for code in sta_list:
            pgd_dict[code] = self.station_timeseries(
                code).pgd_timeseries(t_origin, tau=tau, window=window)
        return pgd_dict

    def ground_displ_timeseries(self, t_origin, sta_list=None,
                                tau=10, window=300):
        if sta_list is None:
            sta_list = self.station_codes()
        ground_displ_dict = dict()
        for code in sta_list:
            ground_displ_dict[code] = self.station_timeseries(
                code).ground_displ_timeseries(t_origin, tau=tau, window=window)
        return ground_displ_dict

    def set_win_offset(self, win_offset):
        win = parse_time(win_offset)
        for ts in self._station_ts:
            ts.win_offset = win


def mw_melgar(pgd_cm, r):
    """

    :param pgd_cm: PGD in cm
    :param r: distance to hipocenter in km
    :return:
    """
    return (np.log10(pgd_cm) + 4.434)/(1.047 - 0.138*np.log10(r))

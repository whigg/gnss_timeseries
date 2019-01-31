from math import sqrt, log10, isfinite
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

    def available_window(self, sta_code=None):
        if self.n_sta == 0:
            return None, None
        index = 0 if sta_code is None else self._code2index[sta_code]
        ts = self._station_ts[index]
        return ts.time_range(), ts.t_oldest

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

    def empty_station_buffer(self, sta_code):
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
            pgd = pgd_dict[code]
            if isfinite(pgd):
                results_dict[code] = (mw_melgar(100*pgd, r), r)
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

    def mw_timeseries_from_pgd(self, hipocenter_coords, t_origin, vel_mask=2,
                               sta_list=None, tau=10, max_distance=800,
                               t_tail=120):
        distance_dict = self._distance_dict(hipocenter_coords,
                                            max_distance=max_distance)
        # times at which each station is reached by the mask
        t_sorted = []
        codes_sorted = []
        for code, r in distance_dict.items():
            t_sorted.append(r/vel_mask)
            codes_sorted.append(code)
        indices = np.argsort(t_sorted)
        t_sorted = np.array([t_origin + t_sorted[k] for k in indices])
        codes_sorted = np.array([codes_sorted[k] for k in indices])
        pgd_dict = self.pgd_timeseries(t_origin, sta_list=sta_list, tau=tau)
        t_min = np.inf
        t_max = -np.inf
        for v in pgd_dict.values():
            aux = v[1][0]
            if aux < t_min:
                t_min = aux
            aux = v[1][-1]
            if aux > t_max:
                t_max = aux
        t_step = 1./self.s_rate
        t = np.arange(t_min, t_max + 0.5*t_step, t_step)
        # todo Terminar ESTO

        print('~'*200)
        print(t.size)
        print('.'*125)
        for pgd in pgd_dict.values():
            print(pgd.size)
        print('~'*200)
        indices_mask = []
        i_aux = 0
        for tau in t_sorted:
            if tau > t[-1]:
                break
            i_aux += np.argmin(np.abs(t[i_aux:] - tau))
            indices_mask.append(i_aux)
        i_mask_0 = indices_mask[0]
        t_mw = (t[i_mask_0:(indices_mask[-1] + int(t_tail*self.s_rate))]
                - t_origin)
        mw = np.zeros_like(t_mw)
        i_mask_curr = i_mask_0
        indices_mask.append(i_mask_0 + mw.size)
        print(indices_mask, t.size, 'qwjfnwpofkw')
        for j in range(1, len(indices_mask)):
            i_mask_next = indices_mask[j]
            for i in range(i_mask_curr, i_mask_next):
                sum_mw = 0.0
                n_ok = 0
                for code in codes_sorted[:j]:
                    if pgd_dict[code][i] > 0.00001:
                        sum_mw += mw_melgar(100*pgd_dict[code][i],
                                            distance_dict[code])
                        n_ok += 1
                mw[i-i_mask_0] = sum_mw/n_ok if n_ok > 0 else np.nan
        return mw, t_mw

    def pgd_timeseries(self, t_origin, sta_list=None, tau=10):
        if sta_list is None:
            sta_list = self.station_codes()
        pgd_dict = dict()
        for code in sta_list:
            pgd_dict[code] = self.station_timeseries(
                code).pgd_timeseries(t_origin, tau=tau)
        return pgd_dict

    def ground_displ_timeseries(self, t_origin, sta_list=None, tau=10):
        if sta_list is None:
            sta_list = self.station_codes()
        ground_displ_dict = dict()
        for code in sta_list:
            ground_displ_dict[code] = self.station_timeseries(
                code).ground_displ_timeseries(t_origin, tau=tau)
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
    return (log10(pgd_cm) + 4.434)/(1.047 - 0.138*log10(r))

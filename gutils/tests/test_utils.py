#!python
# coding=utf-8
from gutils import get_decimal_degrees, interpolate_gps, masked_epoch

from gutils.yo import (
    assign_profiles
)

from gutils.filters import (
    default_filter,
    filter_profile_depth,
    filter_profile_distance,
    filter_profile_number_of_points,
    filter_profile_timeperiod
)

from gutils.ctd import (
    calculate_practical_salinity,
    calculate_density
)

from gutils.slocum import SlocumReader
from gutils.tests import GutilsTestClass, resource

import logging
L = logging.getLogger(__name__)  # noqa


def is_continuous(profiled_dataset):
    last_profile_id = 0
    for i, row in enumerate(profiled_dataset.itertuples()):
        profile_diff = abs(last_profile_id - row.profile)

        if profile_diff == 1:
            last_profile_id = row.profile
        elif profile_diff > 1:
            print(
                "Inconsistency @: %d, Last Profile: %d, Current: %d"
                % (i, last_profile_id, row.profile)
            )
            return False

    return True


ctd_filepath = resource(
    'slocum',
    'bass-test-ascii',
    'rt',
    'ascii',
    'usf_bass_2016_253_0_6_sbd.dat'
)


class TestFindProfile(GutilsTestClass):

    def setUp(self):
        super(TestFindProfile, self).setUp()

        sr = SlocumReader(ctd_filepath)
        self.df = sr.standardize()

        self.profiled_dataset = assign_profiles(
            self.df,
            tsint=10
        )

    def test_find_profile(self):
        assert len(self.profiled_dataset) != 0
        assert len(self.profiled_dataset) == len(self.df)
        assert len(self.profiled_dataset.profile.dropna().unique()) == 32

        # import matplotlib.dates as mpd
        # df = self.profiled_dataset.copy()
        # df['z'] = df.z.values * -1
        # df['t'] = mpd.date2num(df.t.dt.to_pydatetime())
        # df.plot.scatter(x='t', y='z', c='profile', cmap='viridis')

    def test_extreme_depth_filter(self):
        # This should filter all profiles with at least 1m of depth
        meters = 1
        # Compute manually
        df = self.profiled_dataset.copy()
        df = df[df.z > meters]
        df = df[~df.z.isnull()]
        profs = df.profile[~df.profile.isnull()].unique()

        fdf, _, _ = filter_profile_depth(
            self.profiled_dataset,
            below=meters
        )
        profiles = fdf.profile.unique()
        assert len(profiles) == profs.size
        assert is_continuous(fdf) is True

        # No profiles are deeper than 10000m
        fdf, _, _ = filter_profile_depth(
            self.profiled_dataset,
            below=10000
        )
        profiles = fdf.profile.unique()
        assert len(profiles) == 0
        assert is_continuous(fdf) is True
        assert fdf.empty

    def test_filter_profile_timeperiod(self):
        # This should filter all profiles that are less than 300 seconds long
        fdf, _, _ = filter_profile_timeperiod(
            self.profiled_dataset,
            timespan_condition=300
        )
        profiles = fdf.profile.unique()
        assert len(profiles) == 22
        assert is_continuous(fdf) is True

        # This should filter all profiles that are less than 0 seconds (none of them)
        fdf, _, _ = filter_profile_timeperiod(
            self.profiled_dataset,
            timespan_condition=0
        )
        profiles = fdf.profile.unique()
        assert len(profiles) == 32
        assert is_continuous(fdf) is True

        # This should filter all profiles that are less than 10000 seconds (all of them)
        fdf, _, _ = filter_profile_timeperiod(
            self.profiled_dataset,
            timespan_condition=10000
        )
        profiles = fdf.profile.unique()
        assert len(profiles) == 0
        assert is_continuous(fdf) is True
        assert fdf.empty

    def test_filter_profile_distance(self):
        # This should filter all profiles that are not 0m in depth distance
        fdf, _, _ = filter_profile_distance(
            self.profiled_dataset,
            distance_condition=0
        )
        profiles = fdf.profile.unique()
        assert len(profiles) == 32
        assert is_continuous(fdf) is True

        # This should filter all profiles that are not 10000m in depth distance (all of them)
        fdf, _, _ = filter_profile_distance(
            self.profiled_dataset,
            distance_condition=10000
        )
        profiles = fdf.profile.unique()
        assert len(profiles) == 0
        assert is_continuous(fdf) is True
        assert fdf.empty

    def test_filter_profile_number_of_points(self):
        # This should filter all profiles that don't have any points
        fdf, _, _ = filter_profile_number_of_points(
            self.profiled_dataset,
            points_condition=0
        )
        profiles = fdf.profile.unique()
        assert len(profiles) == 32
        assert is_continuous(fdf) is True

        # This should filter all profiles that don't have at least 100000 points (all of them)
        fdf, _, _ = filter_profile_number_of_points(
            self.profiled_dataset,
            points_condition=100000
        )
        profiles = fdf.profile.unique()
        assert len(profiles) == 0
        assert is_continuous(fdf) is True
        assert fdf.empty

        # Compute manually the longest profile in terms of points and filter by that
        # so we only have one as a result
        max_points = self.profiled_dataset.groupby('profile').size().max()
        fdf, _, _ = filter_profile_number_of_points(
            self.profiled_dataset,
            points_condition=max_points
        )
        profiles = fdf.profile.unique()
        assert len(profiles) == 1
        assert is_continuous(fdf) is True
        assert profiles == [0]  # Make sure it reindexed to the "0" profile

        # Now if we add one we should get no profiles
        fdf, _, _ = filter_profile_number_of_points(
            self.profiled_dataset,
            points_condition=max_points + 1
        )
        profiles = fdf.profile.unique()
        assert len(profiles) == 0
        assert is_continuous(fdf) is True
        assert fdf.empty

    def test_default_filter_composite(self):
        fdf, _, _ = filter_profile_depth(self.profiled_dataset)
        fdf, _, _ = filter_profile_number_of_points(fdf)
        fdf, _, _ = filter_profile_timeperiod(fdf)
        fdf, _, _ = filter_profile_distance(fdf)
        profiles = fdf.profile.unique()
        assert len(profiles) == 32
        assert is_continuous(fdf) is True

        default_df, _ = default_filter(self.profiled_dataset)
        # Make sure the default filter works as intended
        assert len(profiles) == 32
        assert is_continuous(default_df) is True
        assert default_df.equals(fdf)

        # import matplotlib.dates as mpd
        # import matplotlib.pyplot as plt
        # df = default_df.copy()
        # df['z'] = df.z.values * -1
        # df['t'] = mpd.date2num(df.t.dt.to_pydatetime())
        # df.plot.scatter(x='t', y='z', c='profile', cmap='viridis')
        # plt.show()


class TestInterpolateGPS(GutilsTestClass):

    def setUp(self):
        super(TestInterpolateGPS, self).setUp()

        sr = SlocumReader(ctd_filepath)
        self.df = sr.standardize()

    def test_interpolate_gps(self):
        est_lat, est_lon = interpolate_gps(
            timestamps=masked_epoch(self.df.t),
            latitude=self.df.y,
            longitude=self.df.x
        )
        assert len(est_lat) == len(est_lon)
        assert len(est_lat) == self.df.y.size
        assert len(est_lon) == self.df.x.size


class TestSalinity(GutilsTestClass):

    def test_practical_salinity(self):
        sr = SlocumReader(ctd_filepath)
        salinity = calculate_practical_salinity(
            sr.data.sci_water_cond,
            sr.data.sci_water_temp,
            sr.data.sci_water_pressure,
        )
        assert sr.data.sci_m_present_time.size == salinity.size


class TestDensity(GutilsTestClass):

    def test_density(self):
        sr = SlocumReader(ctd_filepath)
        df = sr.standardize()

        salinity = calculate_practical_salinity(
            sr.data.sci_water_cond.values,
            sr.data.sci_water_temp.values,
            sr.data.sci_water_pressure.values,
        )
        assert sr.data.sci_m_present_time.size == salinity.size

        est_lat, est_lon = interpolate_gps(
            timestamps=masked_epoch(df.t),
            latitude=df.y,
            longitude=df.x
        )

        density = calculate_density(
            sr.data.sci_water_temp.values,
            sr.data.sci_water_pressure.values,
            salinity,
            est_lat,
            est_lon
        )
        assert sr.data.sci_m_present_time.size == density.size


class TestUtility(GutilsTestClass):

    def test_decimal_degrees(self):
        decimal_degrees = get_decimal_degrees(-8330.567)
        self.assertEqual(
            decimal_degrees,
            -83.50945
        )

        decimal_degrees = get_decimal_degrees(3731.9404)
        self.assertEqual(
            decimal_degrees,
            37.53234
        )

        decimal_degrees = get_decimal_degrees(10601.6986)
        self.assertEqual(
            decimal_degrees,
            106.02831
        )

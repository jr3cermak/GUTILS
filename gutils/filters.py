#!python
# coding=utf-8
import os
import pandas as pd
import numpy as np

from gutils.yo import assign_profiles

import logging
L = logging.getLogger(__name__)


def default_filter(dataset):
    dataset, rm_depth,    _ = filter_profile_depth(dataset, reindex=False)
    dataset, rm_points,   _ = filter_profile_number_of_points(dataset, reindex=False)
    dataset, rm_time,     _ = filter_profile_timeperiod(dataset, reindex=False)
    dataset, rm_distance, _ = filter_profile_distance(dataset, reindex=True)
    total_filtered = rm_depth + rm_points + rm_time + rm_distance
    return dataset, total_filtered


def filter_profiles(dataset, conditional, reindex=True):
    """Filters out profiles that do not meet some criteria

    Returns the filtered set of profiles
    """
    before = len(dataset.profile.unique())
    filtered = dataset.groupby('profile').filter(conditional).copy()
    after = len(filtered.profile.unique())
    # Re-index the profiles
    if reindex is True:
        f, _ = pd.factorize(filtered.profile)
        filtered.loc[:, 'profile'] = f.astype('int32')  # Avoid the int64 dtype

    return filtered, before - after


def filter_profile_depth(dataset, below=None, reindex=True):
    """Filters out profiles that are not completely below a certain depth (Default: 1m). This is
    depth positive down.

    Returns a DataFrame with a subset of profiles
    """

    if below is None:
        below = 1

    def conditional(profile):
        return profile.z.max() >= below

    return (*filter_profiles(dataset, conditional, reindex=reindex), below)


def filter_profile_timeperiod(dataset, timespan_condition=None, reindex=True):
    """Filters out profiles that do not span a specified number of seconds
    (Default: 10 seconds)

    Returns a DataFrame with a subset of profiles
    """
    if timespan_condition is None:
        timespan_condition = 10

    def conditional(profile):
        timespan = profile.t.max() - profile.t.min()
        return timespan >= pd.Timedelta(timespan_condition, unit='s')

    return (*filter_profiles(dataset, conditional, reindex=reindex), timespan_condition)


def filter_profile_distance(dataset, distance_condition=None, reindex=True):
    """Filters out profiles that do not span a specified vertical distance
    (Default: 1m)

    Returns a DataFrame with a subset of profiles
    """
    if distance_condition is None:
        distance_condition = 1

    def conditional(profile):
        distance = abs(profile.z.max() - profile.z.min())
        return distance >= distance_condition

    return (*filter_profiles(dataset, conditional, reindex=reindex), distance_condition)


def filter_profile_number_of_points(dataset, points_condition=None, reindex=True):
    """Filters out profiles that do not have a specified number of points
    (Default: 3 points)

    Returns a DataFrame with a subset of profiles
    """
    if points_condition is None:
        points_condition = 3

    def conditional(profile):
        return len(profile) >= points_condition

    return (*filter_profiles(dataset, conditional, reindex=reindex), points_condition)


def process_dataset(file,
                    reader_class,
                    *,
                    tsint=None,
                    filter_z=None,
                    filter_points=None,
                    filter_time=None,
                    filter_distance=None,
                    z_axis_method=1,
                    **extra_kwargs):

    # Check filename
    if file is None:
        raise ValueError('Must specify path to combined ASCII file')

    try:
        reader = reader_class(file)
        data = reader.standardize(z_axis_method=z_axis_method)

        # The data section is also passed to extras processing.
        # In certain circumstances, the dimension for certain
        # variables are reassigned with the extras dimension.
        extras, data = reader.extras(data, **extra_kwargs)

        if 'z' not in data.columns:
            L.warning("No Z axis found - Skipping {}".format(file))
            return None, None, None

        if 't' not in data.columns:
            L.warning("No T axis found - Skipping {}".format(file))
            return None, None, None

        # Find profile breaks
        profiles = assign_profiles(data, tsint=tsint)
        # Shortcut for empty dataframes
        if profiles is None:
            return None, None, None

        # Filter data
        original_profiles = len(profiles.profile.unique())
        filtered, rm_depth,    did_depth    = filter_profile_depth(profiles, below=filter_z, reindex=False)
        filtered, rm_points,   did_points   = filter_profile_number_of_points(filtered, points_condition=filter_points, reindex=False)
        filtered, rm_time,     did_time     = filter_profile_timeperiod(filtered, timespan_condition=filter_time, reindex=False)
        filtered, rm_distance, did_distance = filter_profile_distance(filtered, distance_condition=filter_distance, reindex=True)
        total_filtered = rm_depth + rm_points + rm_time + rm_distance
        L.info(
            (
                'Filtered {}/{} profiles from {}'.format(total_filtered, original_profiles, os.path.basename(file)),
                'Depth ({}m): {}'.format(did_depth, rm_depth),
                'Points ({}): {}'.format(did_points, rm_points),
                'Time ({}s): {}'.format(did_time, rm_time),
                'Distance ({}m): {}'.format(did_distance, rm_distance),
            )
        )

        # Downscale profile
        # filtered['profile'] = pd.to_numeric(filtered.profile, downcast='integer')
        filtered['profile'] = filtered.profile.astype('int32')
        # Profiles are 1-indexed, so add one to each
        filtered['profile'] = filtered.profile.values + 1

        # TODO: Backfill U/V?
        # TODO: Backfill X/Y?

        # Combine extra data, which has a time index already
        # This puts the profile information into the extras
        # dataframe
        if not extras.empty:
            try:
                merge = pd.merge_asof(
                    extras,
                    filtered[['t', 'profile']],
                    left_index=True,
                    right_index=False,
                    right_on='t',
                    direction='nearest',
                    tolerance=pd.Timedelta(minutes=10)
                ).set_index(extras.index)
                extras['profile'] = merge.profile.ffill()

                # To have consistent netCDF files, empty "extras" variables need to exist
                # in for each valid profile that was calculated above into "filtered".
                profile_list = set(filtered['profile'].unique())
                extras_list = set(extras['profile'].unique().astype('int32'))
                profiles_to_add = profile_list.difference(extras_list)
                if profiles_to_add:
                    first_t_in_profiles = filtered.groupby(by=["profile"]).min()['t']
                    for profile_to_add in profiles_to_add:
                        empty_df = pd.DataFrame([[np.nan] * len(extras.columns)], columns=extras.columns)
                        empty_df['profile'] = profile_to_add
                        empty_df['pseudogram_time'] = first_t_in_profiles[profile_to_add]
                        empty_df.set_index('pseudogram_time', inplace=True)
                        extras = pd.concat([extras, empty_df], sort=True)

            except BaseException as e:
                L.error(f"Could not merge 'extras' data, skipping: {e}")

    except ValueError as e:
        L.exception('{} - Skipping'.format(e))
        raise

    return filtered, extras, reader.mode

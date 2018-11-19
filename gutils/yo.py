#!python
# coding=utf-8
import numpy as np
import pandas as pd

from gutils import (
    masked_epoch,
    boxcar_smooth_dataset
)

import logging
L = logging.getLogger(__name__)


def calculate_delta_depth(interp_data):
    """ Figure out when the interpolated Z data turns a corner
    """
    delta_depth = np.diff(interp_data)
    delta_depth[delta_depth <= 0] = -1
    delta_depth[delta_depth >= 0] = 1
    delta_depth = boxcar_smooth_dataset(delta_depth, 2)
    delta_depth[delta_depth <= 0] = -1
    delta_depth[delta_depth >= 0] = 1
    return delta_depth


def assign_profiles(df, tsint=1):
    profile_df = df.copy()
    profile_df['profile'] = np.nan  # Fill profile with nans
    tmp_df = df.copy()

    if tsint is None:
        tsint = 1

    # Make 't' epochs and not a DateTimeIndex
    tmp_df['t'] = masked_epoch(tmp_df.t)
    # Set negative depth values to NaN
    tmp_df.loc[tmp_df.z <= 0, 'z'] = np.nan

    # Remove any rows where time or z is NaN
    tmp_df = tmp_df.dropna(subset=['t', 'z'], how='any')

    if len(tmp_df) < 2:
        return None

    # Create the fixed timestamp array from the min timestamp to the max timestamp
    # spaced by tsint intervals
    ts = np.arange(tmp_df.t.min(), tmp_df.t.max(), tsint)
    # Stretch estimated values for interpolation to span entire dataset
    interp_z = np.interp(
        ts,
        tmp_df.t,
        tmp_df.z,
        left=tmp_df.z.iloc[0],
        right=tmp_df.z.iloc[-1]
    )

    del tmp_df

    if len(interp_z) < 2:
        return None

    filtered_z = boxcar_smooth_dataset(interp_z, max(tsint // 2, 1))
    delta_depth = calculate_delta_depth(filtered_z)

    # Find where the depth indexes (-1 and 1) flip
    inflections = np.where(np.diff(delta_depth) != 0)[0]
    # Do we have any profiles?
    if inflections.size < 1:
        return profile_df

    # Prepend a zero at the beginning start the series of profiles
    p_inds = np.insert(inflections, 0, 0)
    # Append the size of the time array to end the series of profiles
    p_inds = np.append(p_inds, ts.size - 1)
    # Zip up neighbors to get the ranges of each profile in interpolated space
    p_inds = list(zip(p_inds[0:-1], p_inds[1:]))
    # Convert the profile indexes into datetime objets
    p_inds = [
        (
            pd.to_datetime(ts[int(p0)], unit='s'),
            pd.to_datetime(ts[int(p1)], unit='s')
        )
        for p0, p1 in p_inds
    ]

    # We have the profiles in interpolated space, now associate this
    # space with the actual data using the datetimes.

    # Iterate through the profile start/stop indices
    for profile_index, (min_time, max_time) in enumerate(p_inds):

        # Get rows between the min and max time
        time_between = profile_df.t.between(min_time, max_time, inclusive=True)

        # Get indexes of the between rows since we can't assign by the range due to NaT values
        ixs = profile_df.loc[time_between].index.tolist()

        # Set the rows profile column to the profile id
        if len(ixs) > 1:
            profile_df.loc[ixs[0]:ixs[-1], 'profile'] = profile_index
        elif len(ixs) == 1:
            profile_df.loc[ixs[0], 'profile'] = profile_index
        else:
            L.debug('No data rows matched the time range of this profile, Skipping.')

    # Remove rows that were not assigned a profile
    # profile_df = profile_df.loc[~profile_df.profile.isnull()]

    return profile_df

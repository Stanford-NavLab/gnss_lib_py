"""Visualization functions for GNSS data.

"""

__authors__ = "D. Knowles"
__date__ = "27 Jan 2022"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.collections import LineCollection

from gnss_lib_py.visualizations.style import *
from gnss_lib_py.navdata.navdata import NavData
from gnss_lib_py.utils.coordinates import add_el_az

def plot_skyplot(navdata, receiver_state,
                 save=False, prefix="", fname=None,
                 add_sv_id_label=True, step = "auto", trim_options=None):
    """Skyplot of satellite positions relative to receiver.

    First adds ``el_sv_deg`` and ``az_sv_deg`` rows to navdata if they
    do not yet exist.

    Breaks up satellites by constellation names in ``gnss_id`` and the
    ``sv_id`` if the row is present in navdata.

    Parameters
    ----------
    navdata : gnss_lib_py.navdata.navdata.NavData
        Instance of the NavData class. Must include ``gps_millis`` as
        well as satellite ECEF positions as ``x_sv_m``, ``y_sv_m``,
        ``z_sv_m``, ``gnss_id`` and ``sv_id``.
    receiver_state : gnss_lib_py.navdata.navdata.NavData
        Either estimated or ground truth receiver position in ECEF frame
        in meters as an instance of the NavData class with the
        following rows: ``x_rx*_m``, ``y_rx*_m``, ``z_rx*_m``, ``gps_millis``.
    save : bool
        Saves figure if true to file specified by ``fname`` or defaults
        to the Results folder otherwise.
    prefix : string
        File prefix to add to filename.
    fname : string or path-like
        Path to save figure to. If not None, ``fname`` is passed
        directly to matplotlib's savefig fname parameter and prefix will
        be overwritten.
    add_sv_id_label : bool
        If the ``sv_id`` row is available, will add SV ID label near the
        satellite trail.
    step : int or "auto"
        Skyplot plotting is sped up by only plotting a portion of the
        satellite trajectories. If default is set to "auto" then it will
        plot a maximum of 50 points across each satellite trajectory. If
        the step variable is set to a positive integer ``n`` then only
        every nth point will be used in the trajectory. Setting the
        steps variable to 1 will plot every satellite trajectory point
        and may be slow to plot.
    trim_options : None or dict
        The ``trim_options`` variables gives control for line segments
        being trimmed between satellite points. For example, if 24 hours
        of a satellite is plotted, often the satellite will come in and
        out of view and the segment between when it was lost from view
        and when the satellite comes back in view should be trimmed.
        If trim_options is set to the default of None, then the default
        is set of trimming according to az_and_el and gps_millis. The
        current options for the trim_options dictionary are listed here.
        {"az" : az_limit} means that if at two timesteps the azimuth
        difference in degrees is greater than az_limit, then the line
        segment will be trimmed.
        {"az_and_el" : (az_limit,el_limit)} means that if at two
        timesteps the azimuth difference in degrees is greater than
        az_limit as well as the average of the elevation angle across
        the two timesteps is less than el_limit in degrees, then the
        line segment will be trimmed. The el_limit is because satellites
        near 90 degrees elevation can traverse large amounts of degrees
        in azimuth in a valid trajectory but at low elevations should
        not have such large azimuth changes quickly.
        {"gps_millis",gps_millis_limit} means that line segments will be
        trimmed if the milliseconds between the two points is larger
        than the gps_millis_limit. This option only works if the
        gps_millis row is included in the ``navdata`` variable input.
        Default options for the trim options are :code:`"az_and_el" : (15.,30.)`
        and :code:`"gps_millis" : 3.6E6`.

    Returns
    -------
    fig : matplotlib.pyplot.figure
        Figure object of skyplot.

    """

    if not isinstance(navdata,NavData):
        raise TypeError("first arg to plot_skyplot "\
                          + "must be a NavData object.")

    if not isinstance(prefix, str):
        raise TypeError("Prefix must be a string.")

    # add elevation and azimuth data.
    add_el_az(navdata, receiver_state, inplace=True)

    # create new figure
    fig = plt.figure(figsize=(6,4.5))
    axes = fig.add_subplot(111, projection='polar')

    navdata = navdata.copy()
    navdata["az_sv_rad"] = np.radians(navdata["az_sv_deg"])
    # remove SVs below horizon
    navdata = navdata.where("el_sv_deg",0,"geq")
    # remove np.nan values caused by potentially faulty data
    navdata = navdata.where("az_sv_rad",np.nan,"neq")
    navdata = navdata.where("el_sv_deg",np.nan,"neq")

    for c_idx, constellation in enumerate(sort_gnss_ids(np.unique(navdata["gnss_id"]))):
        const_subset = navdata.where("gnss_id",constellation)
        color = "C" + str(c_idx % len(STANFORD_COLORS))
        cmap = new_cmap(to_rgb(color))
        marker = MARKERS[c_idx % len(MARKERS)]
        const_label_created = False

        # iterate through each satellite
        for sv_name in np.unique(const_subset["sv_id"]):
            sv_subset = const_subset.where("sv_id",sv_name)

            # only plot ~ 50 points for each sat to decrease time
            # it takes to plot these line collections if step == "auto"
            if isinstance(step,str) and step == "auto":
                step = max(1,int(len(sv_subset)/50.))
            elif isinstance(step, int):
                step = max(1,step)
            else:
                raise TypeError("step varaible must be 'auto' or int")
            points = np.array([np.atleast_1d(sv_subset["az_sv_rad"])[::step],
                               np.atleast_1d(sv_subset["el_sv_deg"])[::step]]).T
            points = np.reshape(points,(-1, 1, 2))
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(0,len(segments))

            if trim_options is None:
                trim_options = {
                                "az_and_el" : (15.,30.),
                                "gps_millis" : 3.6E6,
                                }
            plotted_idxs = np.array([True] * len(segments))

            if "az" in trim_options and len(segments) > 2:
                # ignore segments that cross more than az_limit degrees
                # in azimuth between timesteps
                az_limit = np.radians(trim_options["az"])
                az_idxs = ~((np.abs(np.diff(np.unwrap(segments[:,:,0]))) >= az_limit)[:,0])
                plotted_idxs = np.bitwise_and(plotted_idxs, az_idxs)
            if "az_and_el" in trim_options and len(segments) > 2:
                # ignore segments that cross more than az_limit degrees
                # in azimuth between timesteps and are at an elevation
                # less than el_limit degrees.
                # These satellites are assumed to be the satellites
                # coming in and out of view in a later part of the orbit
                az_limit = np.radians(trim_options["az_and_el"][0])
                el_limit = trim_options["az_and_el"][1]
                az_and_el_idxs = ~(((np.abs(np.diff(np.unwrap(segments[:,:,0]))) >= az_limit)[:,0]) \
                                 & (np.mean(segments[:,:,1],axis=1) <= el_limit))
                plotted_idxs = np.bitwise_and(plotted_idxs, az_and_el_idxs)
            if "gps_millis" in trim_options and "gps_millis" in sv_subset.rows \
                and len(segments) > 2:
                # ignore segments if there is more than gps_millis_limit
                # milliseconds between the time segments
                gps_millis_limit = trim_options["gps_millis"]

                all_times = np.atleast_2d(sv_subset["gps_millis"][::step]).T
                point_times = np.concatenate([all_times[:-1],all_times[1:]],
                                              axis=1)
                gps_millis_idxs = (np.abs(np.diff(point_times)) <= gps_millis_limit)[:,0]
                plotted_idxs = np.bitwise_and(plotted_idxs, gps_millis_idxs)

            segments = segments[list(plotted_idxs)]

            local_coord = LineCollection(segments, cmap=cmap,
                            norm=norm, linewidths=(4,),
                            array = range(len(segments)))
            axes.add_collection(local_coord)
            if not const_label_created:
                # plot with label
                axes.plot(np.atleast_1d(sv_subset["az_sv_rad"])[-1],
                          np.atleast_1d(sv_subset["el_sv_deg"])[-1],
                          c=color, marker=marker, markersize=8,
                    label=get_label({"gnss_id":constellation}))
                const_label_created = True
            else:
                # plot without label
                axes.plot(np.atleast_1d(sv_subset["az_sv_rad"])[-1],
                          np.atleast_1d(sv_subset["el_sv_deg"])[-1],
                          c=color, marker=marker, markersize=8)
            if add_sv_id_label:
                # offsets move label to the right of marker
                az_offset = 3.*np.radians(np.cos(np.atleast_1d(sv_subset["az_sv_rad"])[-1]))
                el_offset = -3.*np.sin(np.atleast_1d(sv_subset["az_sv_rad"])[-1])
                axes.text(np.atleast_1d(sv_subset["az_sv_rad"])[-1] \
                          + az_offset,
                          np.atleast_1d(sv_subset["el_sv_deg"])[-1] \
                          + el_offset,
                          str(int(sv_name)),
                          )

    # updated axes for skyplot graph specifics
    axes.set_theta_zero_location('N')
    axes.set_theta_direction(-1)
    axes.set_yticks(range(0, 60+10, 30))    # Define the yticks
    axes.set_yticklabels(['',r'$30\degree$',r'$60\degree$'])
    axes.set_ylim(90,0)

    handles, _ = axes.get_legend_handles_labels()
    if len(handles) > 0:
        axes.legend(loc="upper left", bbox_to_anchor=(1.05, 1),
                   title=get_label({"constellation":"constellation"}))

    fig.set_layout_engine(layout='tight')

    if save: # pragma: no cover
        save_figure(fig, "skyplot", prefix=prefix, fnames=fname)
    return fig

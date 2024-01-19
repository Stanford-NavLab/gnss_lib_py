"""Visualization functions for GNSS data.

"""

__authors__ = "D. Knowles"
__date__ = "27 Jan 2022"

import os
import pathlib
from math import floor
from multiprocessing import Process

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from gnss_lib_py.visualizations import style
import gnss_lib_py.utils.file_operations as fo
from gnss_lib_py.navdata.navdata import NavData
from gnss_lib_py.navdata.operations import find_wildcard_indexes

def plot_map(*args, sections=0, save=False, prefix="",
             fname=None, width=730, height=520, **kwargs):
    """Map lat/lon trajectories on map.

    By increasing the ``sections`` parameter, it is possible to output
    multiple zoom sections of the trajectories to see finer details.

    Parameters
    ----------
    *args : gnss_lib_py.navdata.navdata.NavData
        Tuple of gnss_lib_py.navdata.navdata.NavData objects. The
        NavData objects should include row names for both latitude and
        longitude in the form of ```lat_*_deg`` and ``lon_*_deg``.
        Must also include ``gps_millis`` if sections >= 2.
    sections : int
        Number of zoomed in sections to make of data. Will only output
        additional plots if sections >= 2. Creates sections by equal
        timestamps using the ``gps_millis`` row.
    save : bool
        Save figure if true. Defaults to saving the figure in the
        Results folder.
    prefix : string
        File prefix to add to filename.
    fname : string or path-like
        Path to save figure to. If not None, ``fname`` is passed
        directly to matplotlib's savefig fname parameter and prefix will
        be overwritten.
    width : int
        Figure width in pixels.
    height : int
        Figure height in pixels.
    mapbox_style : str
        Can optionally be included as one of the ``**kwargs``
        Free options include ``open-street-map``, ``white-bg``,
        ``carto-positron``, ``carto-darkmatter``, ``stamen-terrain``,
        ``stamen-toner``, and ``stamen-watercolor``.

    Returns
    -------
    figs : single or list of plotly.graph_objects.Figure
        Returns single plotly.graph_objects.Figure object if sections is
        <= 1, otherwise returns list of Figure objects containing full
        trajectory as well as zoomed in sections.

    """

    figure_df = None        # plotly works best passing in DataFrame
    color_discrete_map = {} # discrete color map

    for idx, traj_data in enumerate(args):
        if not isinstance(traj_data, NavData):
            raise TypeError("Input(s) to plot_map() must be of type " \
                          + "NavData.")

        # check for lat/lon indexes
        traj_idxs = find_wildcard_indexes(traj_data,
                    wildcards=["lat_*_deg","lon_*_deg"], max_allow=1,
                    excludes=[["lat_sigma_*_deg"],["lon_sigma_*_deg"]])

        label_name = style.get_label({"":"_".join((traj_idxs["lat_*_deg"][0].split("_"))[1:-1])})

        data = {"latitude" : traj_data[traj_idxs["lat_*_deg"][0]],
                "longitude" : traj_data[traj_idxs["lon_*_deg"][0]],
                "Trajectory" : [label_name] * len(traj_data),
                }
        if sections >= 2:
            traj_data.in_rows("gps_millis")
            data["gps_millis"] = traj_data["gps_millis"]
        traj_df = pd.DataFrame.from_dict(data)
        color_discrete_map[label_name] = \
                            style.STANFORD_COLORS[idx % len(style.STANFORD_COLORS)]
        if figure_df is None:
            figure_df = traj_df
        else:
            figure_df = pd.concat([figure_df,traj_df])

    zoom, center = _zoom_center(lats=figure_df["latitude"].to_numpy(),
                                lons=figure_df["longitude"].to_numpy(),
                                width_to_height=float(0.9*width)/height)

    fig = px.scatter_mapbox(figure_df,
                            lat="latitude",
                            lon="longitude",
                            color="Trajectory",
                            color_discrete_map=color_discrete_map,
                            zoom=zoom,
                            center=center,
                            width = width,
                            height = height,
                            )
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_layout(**kwargs)

    if sections <= 1:
        if save: # pragma: no cover
            _save_plotly(fig, titles="map", prefix=prefix, fnames=fname,
                         width=width, height=height)
        return fig

    figs = [fig]
    titles = ["map_full"]
    # break into zoom section of figures
    time_groups = np.array_split(np.sort(figure_df["gps_millis"].unique()),sections)
    for time_idx, time_group in enumerate(time_groups):
        zoomed_df = figure_df[(figure_df["gps_millis"] >= min(time_group)) \
                            & (figure_df["gps_millis"] <= max(time_group))]

        # calculate new zoom and center based on partial data
        zoom, center = _zoom_center(lats=zoomed_df["latitude"].to_numpy(),
                                    lons=zoomed_df["longitude"].to_numpy(),
                                    width_to_height=float(0.9*width)/height)
        fig = px.scatter_mapbox(figure_df,
                                lat="latitude",
                                lon="longitude",
                                color="Trajectory",
                                color_discrete_map=color_discrete_map,
                                zoom=zoom,
                                center=center,
                                width = width,
                                height = height,
                                )
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        fig.update_layout(**kwargs)

        figs.append(fig)
        titles.append("map_section_" + str(time_idx + 1))

    if save: # pragma: no cover
        _save_plotly(figs, titles=titles, prefix=prefix, fnames=fname,
                     width=width, height=height)
    return figs


def _zoom_center(lats, lons, width_to_height = 1.25):
    """Finds optimal zoom and centering for a plotly mapbox.

    Assumed to use Mercator projection.

    Temporary solution copied from stackoverflow [1]_ and awaiting
    official implementation [2]_.

    Parameters
    --------
    lons: array-like,
        Longitude component of each location.
    lats: array-like
        Latitude component of each location.
    width_to_height: float, expected ratio of final graph's width to
        height, used to select the constrained axis.

    Returns
    -------
    zoom: float
        Plotly zoom parameter from 1 to 20.
    center: dict
        Position with 'lon' and 'lat' keys for cetner of map.

    References
    ----------
    .. [1] Richie V. https://stackoverflow.com/a/64148305/12995548.
    .. [2] https://github.com/plotly/plotly.js/issues/3434

    """

    maxlon, minlon = max(lons), min(lons)
    maxlat, minlat = max(lats), min(lats)
    center = {
        'lon': round((maxlon + minlon) / 2, 6),
        'lat': round((maxlat + minlat) / 2, 6)
    }

    # longitudinal range by zoom level (20 to 1)
    # in degrees, if centered at equator
    lon_zoom_range = np.array([
        0.0007, 0.0014, 0.003, 0.006, 0.012, 0.024, 0.048, 0.096,
        0.192, 0.3712, 0.768, 1.536, 3.072, 6.144, 11.8784, 23.7568,
        47.5136, 98.304, 190.0544, 360.0
    ])

    # assumed Mercator projection
    margin = 2.5
    height = (maxlat - minlat) * margin * width_to_height
    width = (maxlon - minlon) * margin
    lon_zoom = np.interp(width , lon_zoom_range, range(20, 0, -1))
    lat_zoom = np.interp(height, lon_zoom_range, range(20, 0, -1))
    zoom = floor(min(lon_zoom, lat_zoom))
    # zoom level higher than 18 won't load load properly as of June 2023
    zoom = min(zoom, 18)

    return zoom, center

def _save_plotly(figures, titles=None, prefix="", fnames=None,
                 width=730, height=520): # pragma: no cover
    """Saves figures to file.

    Parameters
    ----------
    figures : single or list of plotly.graph_objects.Figure objects to
        be saved.
    titles : string, path-like or list of strings
        Titles for all plots.
    prefix : string
        File prefix to add to filename.
    fnames : single or list of string or path-like
        Path to save figure to. If not None, ``fname`` is passed
        directly to plotly's write_image file parameter and prefix will
        be overwritten.
    width : int
        Figure width in pixels.
    height : int
        Figure height in pixels.

    """

    if isinstance(figures, go.Figure):
        figures = [figures]
    if isinstance(titles,str) or titles is None:
        titles = [titles]
    if isinstance(fnames, (str, pathlib.Path)) or fnames is None:
        fnames = [fnames]

    for fig_idx, figure in enumerate(figures):

        if (len(fnames) == 1 and fnames[0] is None) \
            or fnames[fig_idx] is None:
            # create results folder if it does not yet exist.
            log_path = os.path.join(os.getcwd(),"results",fo.TIMESTAMP)
            fo.make_dir(log_path)

            # make name path friendly
            title = titles[fig_idx]
            title = title.replace(" ","_")
            title = title.replace(".","")

            if prefix != "" and not prefix.endswith('_'):
                prefix += "_"
            fname = os.path.join(log_path, prefix + title \
                                                  + ".png")
        else:
            fname = fnames[fig_idx]

        while True:
            # sometimes writing a plotly image hanges for an unknown
            # reason. Hence, we call write_image in a process that is
            # automatically terminated after 180 seconds if nothing
            # happens.
            process = Process(target=_write_plotly,
                               name="write_plotly",
                               args=(figure,fname,width,height))
            process.start()

            process.join(180)
            if process.is_alive():
                process.terminate()
                process.join()
                continue
            break

def _write_plotly(figure, fname, width, height): # pragma: no cover
    """Saves figure to file.

    Automatically zooms out if plotly throws a ValueError when trying
    to zoom in too much.

    Parameters
    ----------
    figure : plotly.graph_objects.Figure
        Object to save.
    fname : string or path-like
        Path to save figure to.
    width : int
        Figure width in pixels.
    height : int
        Figure height in pixels.

    """

    while True:
        try:
            figure.write_image(fname,
                               width = width,
                               height = height,
                               )
            break
        except ValueError as error:
            figure.layout.mapbox.zoom -= 1
            if figure.layout.mapbox.zoom < 1:
                print(error)
                break

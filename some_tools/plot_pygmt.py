"""
This module is a wrapper for ADAPT-API results into GMT.
Thanks to PyGMT
"""

import yaml
import pygmt
from obspy import UTCDateTime
import numpy as np
import logging
import some_tools.errors as STE
from pathlib import Path

logger = logging.getLogger(__name__)


# ====================================================================
# ================================================  Private functions
def _get_conf(filepath, check_version=True):
    """ Simple function to unpack the YAML configuration file
        configuration file and return as a dict.
    """
    from some_tools import __version__

    # Create dict
    try:
        with open(filepath, "rt") as qfc:
            outDict = yaml.load(qfc, Loader=yaml.FullLoader)
    except KeyError as err:
        raise STE.BadConfigurationFile("Wrong key name/type!")

    # --- Check Versions
    if check_version:
        if __version__.lower() != outDict['some_tools_version']:
            raise STE.BadConfigurationFile("SOME-TOOLS version [%s] and "
                                           "CONFIG version [%s] differs!" %
                                           (__version__,
                                            outDict['some_tools_version']))
    #
    return outDict

# ====================================================================
# ====================================================================
# ====================================================================


# pygmt_conf = _get_conf("config/pygmt_defaults.yml")


# ========================================  PyGMT defaults
pygmt.config(MAP_FRAME_TYPE="plain")
pygmt.config(FORMAT_GEO_MAP="ddd.xx")
pygmt.config(FORMAT_GEO_MAP="ddd.xxF")
# pygmt.config(FONT_ANNOT_PRIMARY="8p,2")  # 0 normal, 1 bold, 2 italic
pygmt.config(FONT_ANNOT_PRIMARY="8p,0")  # 0 normal, 1 bold, 2 italic
pygmt.config(FONT_LABEL="11p,1")
pygmt.config(PROJ_LENGTH_UNIT="c")



DEFAULTMAPLIMIT = [1, 21, 41, 51]
MAP_FIG_WIDTH = 12  # cm
MAP_FIG_HEIGHT = 12  # cm
TRACE_FIG_WIDTH = 15  # cm
TRACE_FIG_HEIGHT = 4  # cm
# =============================================================


def _format_utcstr(utc):
    """ Input must be an UTCDateTime obj """
    utcstr = ("%4d-%02d-%02d %02d:%02d:%06.3f" % (
                                utc.year,
                                utc.month,
                                utc.day,
                                utc.hour,
                                utc.minute,
                                utc.second +
                                utc.microsecond * 10**-6))
    return utcstr


def _centimeter2seconds(xwidth, xmax, seconds):
    """ de """
    return xwidth*seconds/xmax


def _unique_legend(legend_list, key_tag, search_label):
    """ Remove duplicates entries for the same label.
        BOOLEAN return: True if match found
    """
    for dd in legend_list:
        if dd[key_tag].lower() == search_label:
            return True
    #
    return False


def _miniprocess(tr, override=True):
    """ simple and fast processing routine """
    if not override:
        wtr = tr.copy()
    else:
        wtr = tr
    #
    wtr.detrend('demean')
    wtr.detrend('simple')
    # wtr.taper(max_percentage=0.05, type='cosine')
    wtr.filter("bandpass",
               freqmin=1,
               freqmax=20,
               corners=2,
               zerophase=True)
    return wtr


def obspyTrace2GMT(tr,
                   plot_time_marks=False,
                   show=True,
                   uncertainty_center=None,
                   uncertainty_window=None,
                   #
                   big_x_tick_interval=None,
                   small_x_tick_interval=None,
                   big_y_tick_interval=None,
                   small_y_tick_interval=None,
                   #
                   fig_width=TRACE_FIG_WIDTH,
                   fig_height=TRACE_FIG_HEIGHT,
                   store_name=None):
    """Plot obspy trace with GMT renders

    Simple wrap around PyGMT library

    Args:

    Returns:

    """
    t = tr.times()
    xmin = min(t)
    xmax = max(t)
    ymin = min(tr.data)
    ymax = max(tr.data)

    # ================================= Set Frame INTERVALs

    if not big_x_tick_interval:
        xlabelMaj = float((xmax-xmin)/6.0)
    else:
        xlabelMaj = big_x_tick_interval
    #
    if not small_x_tick_interval:
        xlabelMin = float((xmax-xmin)/30.0)
    else:
        xlabelMin = small_x_tick_interval
    #
    if not big_y_tick_interval:
        ylabelMaj = int((ymax-ymin)/10.0)
    else:
        ylabelMaj = big_y_tick_interval
    #
    if not small_y_tick_interval:
        ylabelMin = int((ymax-ymin)/50.0)
    else:
        ylabelMin = small_y_tick_interval

    # =====================================================

    region = [xmin, xmax, ymin, ymax]
    projection = "X%dc/%d" % (fig_width, fig_height)

    frame = ["xa%.1ff%.1f" % (xlabelMaj, xlabelMin),
             "ya%df%d" % (ylabelMaj, ylabelMin),
             "WS", "x+ltime(s)", "y+lcounts"]

    # ------------------  Plot
    # @@@ Layer 0 : Canvas
    fig = pygmt.Figure()
    fig.basemap(region=region, projection=projection, frame=frame)

    # @@@ Layer 1 : Uncertainties
    if plot_time_marks:
        if (isinstance(uncertainty_window, (int, float)) and
           isinstance(uncertainty_center, UTCDateTime)):
            xunc = _centimeter2seconds(fig_width, xmax, uncertainty_window)
            ttm = uncertainty_center - tr.stats.starttime
            fig.plot(
                data=np.array([[ttm, 0, xunc, fig_height+0.2]]),
                style="rc",
                # transparency=50,
                color="220"
            )

    # @@@ Layer 2 : Waveforms
    fig.plot(
        x=t,
        y=tr.data,
        pen="0.7p,black",
    )

    # @@@ Layer 3 : Text
    fig.text(
        text="start: %s" % _format_utcstr(tr.stats.starttime),
        x=(0.85*xmax)-xmin,
        y=ymin + ylabelMaj*1.1,
        # fill="green",
        font="8p,Helvetica,black")

    # @@@ Layer 4 :  Picks
    if plot_time_marks:
        for xx in tr.stats.timemarks:
            # tuple ( UTC, matplotlibDict{})
            ttm = xx[0] - tr.stats.starttime
            try:
                fig.plot(
                    x=[ttm, ttm],
                    y=[ymin + ylabelMaj*1.5, ymax - ylabelMaj*1.5],
                    pen="%fp,%s" % (xx[1]['ms'], xx[1]['color']),
                    straight_line=True,

                    # Set the legend label,
                    # and set the symbol size to be 0.25 cm (+S0.25c) in legend
                    label=f"{xx[1]['label']}+S0.25c",
                )
            except KeyError:
                # No label --> No legend entry
                fig.plot(
                    x=[ttm, ttm],
                    y=[ymin + ylabelMaj*1.5, ymax - ylabelMaj*1.5],
                    pen="%fp,%s" % (xx[1]['ms'], xx[1]['color']),
                    straight_line=True,
                )

        # Make legend
        fig.legend(transparency=20,
                   position="jBL+o0.1c",
                   box="+p1+g250")
    if show:
        fig.show(method="external")

    if isinstance(store_name, str):
        # remember to use extension "*.png - *.pdf"
        logger.info("Storing figure: %s" % store_name)
        fig.savefig(store_name)
    #
    return fig


# def obspyCatalog2GMT(cat,
#                      # region=None  # Automatically set by the function
#                      projection=default_map_projection,
#                      frame=default_map_frame,
#                      expand_map_lon=default_expand_map_lon,
#                      expand_map_lat=default_expand_map_lat,
#                      use_relief=False,
#                      relief_grid_file=None,
#                      use_hillshade=False,
#                      #
#                      big_x_tick_interval=None,
#                      small_x_tick_interval=None,
#                      big_y_tick_interval=None,
#                      small_y_tick_interval=None,
#                      #
#                      magnitude_scale=0.5,  # cm normalized to Mag=1
#                      show=True,
#                      #
#                      fig_width=MAP_FIG_WIDTH,
#                      fig_height=MAP_FIG_HEIGHT,
#                      store_name=None):
#     """ Transform obspy catalog to PyGMT figure
#     """

#     # ======================== Define region
#     lonlist = []
#     latlist = []
#     maglist = []
#     for _xx, _ev in enumerate(cat):
#         ev = _ev.preferred_origin()
#         if not ev:
#             ev = _ev.origins[0]
#         #
#         ev_mag = _ev.preferred_magnitude()
#         if not ev_mag:
#             ev_mag = _ev.magnitudes[0]
#         #
#         lonlist.append(ev.longitude)
#         latlist.append(ev.latitude)
#         maglist.append(ev_mag.mag)
#     #
#     minLon, maxLon = np.min(lonlist) - expand_map_lon, np.max(lonlist) + expand_map_lon
#     minLat, maxLat = np.min(latlist) - expand_map_lat, np.max(latlist) + expand_map_lat
#     minMag, maxMag = np.min(maglist), np.max(maglist)
#     #
#     totev = _xx+1
#     logger.info("Creating map for %d events ..." % totev)
#     logger.info("MAP region: [%09.5f / %09.5f / %09.5f / %09.5f]" %
#                 (minLon, maxLon, minLat, maxLat))

#     region = [minLon, maxLon, minLat, maxLat]

#     # ========================================
#     fig = pygmt.Figure()
#     fig.basemap(region=region, projection=projection, frame=frame)
#     pygmt.makecpt(cmap="gray", series=[200, 4000, 10])

#     # grid
#     if use_relief:
#         if Path(relief_grid_file).exists():
#             logger.warning("Still to implement the loading of grid files!")  #MB: @develop
#             grid = pygmt.datasets.load_earth_relief(resolution="10m", region=region)
#         else:
#             grid = pygmt.datasets.load_earth_relief(resolution="10m", region=region)
#         #
#         if use_hillshade:
#             dgrid = pygmt.grdgradient(grid=grid, radiance=[20, 180])
#             fig.grdimage(grid=dgrid, cmap=True)
#         else:
#             fig.grdimage(grid=grid, cmap=True)
#     #

#     fig.coast(water="skyblue") # land
#     fig.plot(x=lonlist, y=latlist, style="c0.3c", color="white", pen="0.1,black")

#     if show:
#         fig.show(method="external")

#     if isinstance(store_name, str):
#         # remember to use extension "*.png - *.pdf"
#         logger.info("Storing figure: %s" % store_name)
#         fig.savefig(store_name)
#     #
#     return fig

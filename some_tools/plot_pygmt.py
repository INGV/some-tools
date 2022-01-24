"""
This module is a wrapper for ADAPT-API results into GMT.
Thanks to PyGMT
"""

import pygmt
from obspy import UTCDateTime
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ========================================  PyGMT defaults
pygmt.config(MAP_FRAME_TYPE="plain")
pygmt.config(FORMAT_GEO_MAP="ddd.xx")
pygmt.config(FORMAT_GEO_MAP="ddd.xxF")
# pygmt.config(FONT_ANNOT_PRIMARY="8p,2")  # 0 normal, 1 bold, 2 italic
pygmt.config(FONT_ANNOT_PRIMARY="8p,0")  # 0 normal, 1 bold, 2 italic
pygmt.config(FONT_LABEL="11p,1")
pygmt.config(PROJ_LENGTH_UNIT="c")


DEFAULTMAPLIMIT = [1, 21, 41, 51]
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
                   # #
                   # big_x_tick_interval=5,
                   # small_x_tick_interval=1,
                   # big_y_tick_interval=5,
                   # small_y_tick_interval=1,
                   #
                   store_name=None):
    """Plot obspy trace with GMT renders

    Simple wrap around PyGMT library

    Args:

    Returns:

    """
    FIGWIDTH = 15  # cm
    FIGHEIGHT = 4  # cm
    #
    t = tr.times()
    xmin = min(t)
    xmax = max(t)
    ymin = min(tr.data)
    ymax = max(tr.data)

    # ================================= Set Frame INTERVALs

    if not big_x_tick_interval:
        xlabelMaj = float((xmax-xmin)/6.0)
    if not small_x_tick_interval:
        xlabelMin = float((xmax-xmin)/30.0)
    if not big_y_tick_interval:
        ylabelMaj = int((ymax-ymin)/10.0)
    if not small_y_tick_interval:
        ylabelMin = int((ymax-ymin)/50.0)

    # =====================================================

    region = [xmin, xmax, ymin, ymax]
    projection = "X%dc/%d" % (FIGWIDTH, FIGHEIGHT)

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
            xunc = _centimeter2seconds(FIGWIDTH, xmax, uncertainty_window)
            ttm = uncertainty_center - tr.stats.starttime
            fig.plot(
                data=np.array([[ttm, 0, xunc, FIGHEIGHT+0.2]]),
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

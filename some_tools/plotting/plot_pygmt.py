"""
This module is a wrapper for ADAPT-API results into GMT.
Thanks to PyGMT
"""

import pygmt
from obspy import UTCDateTime
import numpy as np


# DEFAULTS for the pygmt.Figure().
pygmt.config(MAP_FRAME_TYPE="plain")
pygmt.config(FORMAT_GEO_MAP="ddd.xx")
pygmt.config(FORMAT_GEO_MAP="ddd.xxF")
# pygmt.config(FONT_ANNOT_PRIMARY="8p,2")  # 0 normal, 1 bold, 2 italic
pygmt.config(FONT_ANNOT_PRIMARY="8p,0")  # 0 normal, 1 bold, 2 italic
pygmt.config(FONT_LABEL="11p,1")
pygmt.config(PROJ_LENGTH_UNIT="c")


DEFAULTMAPLIMIT = [1, 21, 41, 51]


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


def _miniprocess(tr, inplace=True):
    if not inplace:
        wtr = tr.copy()
    else:
        wtr = tr
    #
    wtr.detrend('demean')
    wtr.detrend('simple')
    # wtr.taper(max_percentage=0.05, type='cosine')
    wtr.filter("bandpass",
               freqmin=1,
               freqmax=10,
               corners=2,
               zerophase=True)
    return wtr


def _append_weighters_results(intr, pd, stat,
                              phase_association="P1",
                              index_association=0):
    """ This function append picks to the trace """

    colordict = {
        'valid_obs': "forestgreen@20",   # "greenyellow",
        'outliers':  "orange"
    }
    #
    intr.stats.timemarks = []
    leglst = []
    weight_obj = pd[stat][phase_association][index_association]['weight']
    triage_dict = weight_obj.get_triage_dict()

    if triage_dict['valid_obs']:
        for lp in triage_dict['valid_obs']:
            if _unique_legend(leglst, "label", "valid"):
                intr.stats.timemarks.append(
                    (UTCDateTime(lp[1]),
                     {
                      'color': colordict['valid_obs'],
                      'ms': 1.5})
                )
            else:
                intr.stats.timemarks.append(
                    (UTCDateTime(lp[1]),
                     {'label': "valid",
                      'color': colordict['valid_obs'],
                      'ms': 1.5})
                )
                #
                leglst.append(
                       {'label': "valid",
                        'color': colordict['valid_obs'],
                        'ms': 1.5})

    if triage_dict['outliers']:
        for lp in triage_dict['outliers']:
            if _unique_legend(leglst, "label", "outlier"):
                intr.stats.timemarks.append(
                    (UTCDateTime(lp[1]),
                     {
                      'color': colordict['outliers'],
                      'ms': 1.5})
                )
            else:
                intr.stats.timemarks.append(
                    (UTCDateTime(lp[1]),
                     {'label': 'outlier',
                      'color': colordict['outliers'],
                      'ms': 1.5})
                )
                #
                leglst.append(
                     {'label': 'outlier',
                      'color': colordict['outliers'],
                      'ms': 1.5})

    return intr


def _append_multipick_slice(intr, pd, stat, slicenum=0):
    """ This function append picks to the trace """

    colordict = {
        'AIC': "limegreen@20",   # "greenyellow",
        'FP':  "deepskyblue@20",
        'BK':  "darkorange@20",
        'HOS': "navajowhite2@20"
    }
    #
    # import pdb; pdb.set_trace()
    intr.stats.timemarks = []
    kp = pd.getStatPick(stat)
    for pickertag in kp:
        pickername = pickertag.split("_")[0]
        if pickertag[-3:].lower() == "mp1":
            if isinstance(slicenum, int):
                print("Slice %d - Adding ... %s" % (slicenum, pickername))
                try:
                    intr.stats.timemarks.append(
                        (pd[stat][pickertag][slicenum]['timeUTC_pick'],
                         {'label': pickername,
                          'color': colordict[pickername],
                          'ms': 1.5})
                    )
                except IndexError:
                    print("Sorry, slice %d for %s missing!" % (
                                                    slicenum, pickername))
            elif isinstance(slicenum, str) and slicenum.lower() == "all":
                print("Slice ALL - Adding ... %s" % pickername)
                for dd in pd[stat][pickertag]:
                    intr.stats.timemarks.append(
                        (dd['timeUTC_pick'],
                         {'label': pickername,
                          'color': colordict[pickername],
                          'ms': 1.5})
                        )
            else:
                print("ERROR! slicenum must either be a digit or ")

    return intr


def obspyTrace2GMT(tr,
                   plot_time_marks=False,
                   show=True,
                   uncertainty_center=None,
                   uncertainty_window=None,
                   big_x_tick_interval=5,
                   small_x_tick_interval=1,
                   store_name=None):
    """ Deh, bellissimo!
    """
    FIGWIDTH = 15  # cm
    FIGHEIGHT = 4  # cm
    #
    t = tr.times()
    xmin = min(t)
    xmax = max(t)
    ymin = min(tr.data)
    ymax = max(tr.data)
    #
    ylabelMaj = int((ymax-ymin)/10.0)
    ylabelMin = int((ymax-ymin)/50.0)

    region = [xmin, xmax, ymin, ymax]
    projection = "X%dc/%d" % (FIGWIDTH, FIGHEIGHT)

    frame = ["xa%.1ff%.1f" % (big_x_tick_interval, small_x_tick_interval,),
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
        fig.savefig(store_name)

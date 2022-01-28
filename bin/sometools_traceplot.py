#!/usr/bin/env python

# Use this function to call the time-series plotting routines

import os
import sys
import yaml
from pathlib import Path
from obspy import read, Stream
from some_tools import plot_pygmt as PG

DEFAULTCONFIG = "/Users/matteo/INGV/SOME-tools/config"


def main(inst, **configdict):

    # Waveforms
    inst


    tr = st.select(channel="*Z")[0]
    tr = PG._miniprocess(tr)
    tr.stats.timemarks = []
    # tr.plot()

    # --- Extract Pick
    #https://colorpalettes.net/color-palette-4289/

    #Dark GREEN  TEAL-GREEN   LightGray   LightBrown   DarkBrown   RED
    #002a29      #007a79      #c7cedf     #976f4f      #4d2a16     #cb132d


    predictedarr = get_pick_slice(
                            pd,
                            stat,
                            "^Predicted_",
                            phase_pick_indexnum=0,
                            arrival_order='all')


    if predictedarr:
        predlabel, pretime = predictedarr[0]
        tr.stats.timemarks.append(
            (pretime,
             {'label': predlabel, 'color': 'cyan', 'ms': 1.3})
            )

    if 'P1' in pd.getStatPick(stat):
        tr.stats.timemarks.append(
            (pd.getMatchingPick(
                stat, 'P1', indexnum=0)[0][1]['timeUTC_pick'],
                {'label': 'ADAPT', 'color': 'red', 'ms': 1.3})
            )
        mainpick = pd.getMatchingPick(
                        stat, 'P1', indexnum=0)[0][1]['timeUTC_pick']
        pearly = pd.getMatchingPick(
                        stat, 'P1', indexnum=0)[0][1]['timeUTC_early']
        plate = pd.getMatchingPick(
                        stat, 'P1', indexnum=0)[0][1]['timeUTC_late']

    if 'Seiscomp_P' in pd.getStatPick(stat):
        tr.stats.timemarks.append(
            (pd.getMatchingPick(
                stat, 'Seiscomp_P', indexnum=0)[0][1]['timeUTC_pick'],
                {'label': 'SC3', 'color': 'blue', 'ms': 1.3})
            )


    # Min Plot
    tr.trim(mainpick-10, mainpick+10)
    PG.obspyTrace2GMT(
                tr, plot_time_marks=True,
                uncertainty_window=(plate - pearly),
                uncertainty_center=pearly + ((plate - pearly) / 2.0),
                store_name="_".join([eqid, stat, "Main.pdf"]))

    # Slice-1
    ntr = tr.copy()
    ntr.trim(mainpick-2, mainpick+2)
    PG._append_multipick_slice(ntr, pd, stat, slicenum=0)   # reset time marks
    PG.obspyTrace2GMT(
                ntr, plot_time_marks=True,
                store_name="_".join([eqid, stat, "Slice1.pdf"]))
                # uncertainty_window=(plate - pearly),
                # uncertainty_center=pearly + ((plate - pearly) / 2.0))


    # Slice-2
    ntr = tr.copy()
    ntr.trim(mainpick-2, mainpick+2)
    PG._append_multipick_slice(ntr, pd, stat, slicenum=1)   # reset time marks
    PG.obspyTrace2GMT(
                ntr, plot_time_marks=True,
                store_name="_".join([eqid, stat, "Slice2.pdf"]))
                # uncertainty_window=(plate - pearly),
                # uncertainty_center=pearly + ((plate - pearly) / 2.0))

    # Slice- WEIGHTER
    ntr = tr.copy()
    ntr.trim(mainpick-2, mainpick+2)
    ntr = PG._append_weighters_results(ntr, pd, stat)   # reset time marks
    ntr.stats.timemarks.append(
            (pd.getMatchingPick(
                stat, 'P1', indexnum=0)[0][1]['timeUTC_pick'],
                {'label': 'ADAPT', 'color': 'red', 'ms': 1.3})
            )
    PG.obspyTrace2GMT(
                ntr, plot_time_marks=True,
                store_name="_".join([eqid, stat, "SliceALLL.pdf"]),
                uncertainty_window=(plate - pearly),
                uncertainty_center=pearly + ((plate - pearly) / 2.0))


if __name__ == "main":
    if len(sys.argv) < 2:
        print("USAGE: %s TRACEPATH [GLOBAL-Json]" % Path(sys.arg[0]).name)
        sys.exit()
    #
    try:
        # load user input
        confdict = yaml.load(sys.argv[2], Loader=yaml.FullLoader)

    except IndexError:
        # load default
        confdict = yaml.load(os.sep.join([DEFAULTCONFIG, "plot_config.yml"]),
                             loader=yaml.FullLoader)

    # --- Load data
    stname = Path(sys.arg[1])
    print("... importing:  %s" % stname.absolute())

    main(**confdict)

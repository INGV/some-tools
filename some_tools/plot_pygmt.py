"""
This module is a wrapper for ADAPT-API results into GMT.
Thanks to PyGMT
"""

import yaml
import pygmt
from pathlib import Path, PurePath
import logging
#
import numpy as np
import pandas as pd
import obspy
from obspy import UTCDateTime
# Some Tools related
import some_tools as ST
import some_tools.errors as STE


logger = logging.getLogger(__name__)

KM = 0.001
MT = 1000
DEFAULTSCONFIGFILE = (str(PurePath(ST.__file__).parent) +
                      "/config/pygmt_defaults.yml")


# ====================================================================
# =======================================  Module's Private Functions
def __get_conf__(filepath, check_version=True):
    """ Simple function to unpack the YAML configuration file
        configuration file and return as a dict.
    """
    stver = ST.__version__

    # Create dict
    try:
        with open(filepath, "rt") as qfc:
            outDict = yaml.load(qfc, Loader=yaml.FullLoader)
    except KeyError:
        raise STE.BadConfigurationFile("Wrong key name/type!")

    # --- Check Versions
    if check_version:
        if stver.lower() != outDict['some_tools_version']:
            raise STE.BadConfigurationFile("SOME-TOOLS version [%s] and "
                                           "CONFIG version [%s] differs!" %
                                           (stver,
                                            outDict['some_tools_version']))
    #
    return outDict


# ====================================================================
# ============================================  Module's Import Setup

# Set global PyGMT config
DEFAULTS = __get_conf__(DEFAULTSCONFIGFILE, check_version=True)
pygmt.config(**DEFAULTS['pygmt_config'])

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


class SomeMapping(object):
    """ Base class for geographical plots based on several databases

    Supported database:
        - pandas.DataFrame
        - geopandas.DataFrame
        - obspy.Catalog

    Class Attributes:
        df (pandas.DataFrame)
        grid ()

    """
    def __init__(self, database=None, grid_data=None, config_file=None):
        self.df = None
        self.grid = None
        #
        self.map_region = None
        self.map_projection = None
        self.map_frame = None
        #
        self.sect_region = None
        self.sect_projection = None
        self.sect_frame = None

        if database:
            self._setup_database(database)
        if grid_data:
            self._setup_grid(grid_data)

        self._setup_class_plot_map()
        self._setup_class_plot_sect()

    def _setup_class_plot_sect(self, config_file=None):
        """ This method will modify the following class-attributes:
            sect_region, sect_projection, sect_frame
        """
        if self.df is None or self.df.empty or not isinstance(self.df, pd.DataFrame):
            logger.warning("Missing database! `auto_scale` and `auto_frame` "
                           "options will be ignored!")
            _miss_df_check = True
        else:
            _miss_df_check = False
        #
        if config_file:
            try:
                _tmp_conf = __get_conf__(
                          config_file, check_version=True)["sect_config"]
            except KeyError:
                raise STE.BadConfigurationFile(
                  "Missing `sect_config` key in config file: %s" % config_file)
        else:
            try:
                _tmp_conf = DEFAULTS["sect_config"]
            except KeyError:
                raise STE.BadConfigurationFile(
                  "Missing `sect_config` key in config file: DEFAULT")

        # ---------- Reset Attributes
        logger.warning("Overriding the default SECT plot class-attributes!")
        self.sect_region, self.sect_projection, self.sect_frame = None, None, None

    def _setup_class_plot_map(self, config_file=None):
        """ This method will modify the following class-attributes:
            map_region, map_projection, map_frame
        """
        if self.df is None or self.df.empty or not isinstance(self.df, pd.DataFrame):
            logger.warning("Missing database! `auto_scale` and `auto_frame` "
                           "options will be ignored!")
            _miss_df_check = True
        else:
            _miss_df_check = False
        #
        if config_file:
            try:
                _tmp_conf = __get_conf__(
                          config_file, check_version=True)["map_config"]
            except KeyError:
                raise STE.BadConfigurationFile(
                  "Missing `map_config` key in config file: %s" % config_file)
        else:
            try:
                _tmp_conf = DEFAULTS["map_config"]
            except KeyError:
                raise STE.BadConfigurationFile(
                  "Missing `map_config` key in config file: DEFAULT")

        # ---------- Reset Attributes
        logger.warning("Overriding the default MAP plot class-attributes!")
        self.map_region, self.map_projection, self.map_frame = None, None, None

        # ---------- Auto-Scale
        if _tmp_conf['auto_scale'] and not _miss_df_check:
            minLon = np.float(
                np.min(self.df["LON"]) - _tmp_conf['expand_map_lon'])
            maxLon = np.float(
                np.max(self.df["LON"]) + _tmp_conf['expand_map_lon'])
            minLat = np.float(
                np.min(self.df["LAT"]) - _tmp_conf['expand_map_lat'])
            maxLat = np.float(
                np.max(self.df["LAT"]) + _tmp_conf['expand_map_lat'])
            #
            meanLon = minLon + ((maxLon - minLon)/2)
            meanLat = minLat + ((maxLat - minLat)/2)
        else:
            minLon = np.float(_tmp_conf['map_region'][0])
            maxLon = np.float(_tmp_conf['map_region'][1])
            minLat = np.float(_tmp_conf['map_region'][2])
            maxLat = np.float(_tmp_conf['map_region'][3])
            #
            meanLon = minLon + ((maxLon - minLon)/2)
            meanLat = minLat + ((maxLat - minLat)/2)

        # ---------- Auto-Frame
        if _tmp_conf['auto_frame'] and not _miss_df_check:
            minLon = np.float(
                np.min(self.df["LON"]) - _tmp_conf['expand_map_lon'])
            maxLon = np.float(
                np.max(self.df["LON"]) + _tmp_conf['expand_map_lon'])
            minLat = np.float(
                np.min(self.df["LAT"]) - _tmp_conf['expand_map_lat'])
            maxLat = np.float(
                np.max(self.df["LAT"]) + _tmp_conf['expand_map_lat'])
            #
            _ax = np.abs(np.float((maxLon-minLon)/6.0))
            _fx = np.abs(np.float((maxLon-minLon)/30.0))
            _ay = np.abs(np.float((maxLat-minLat)/6.0))
            _fy = np.abs(np.float((maxLat-minLat)/30.0))
        else:
            _ax = _tmp_conf['map_frame']['big_x_tick_interval']
            _fx = _tmp_conf['map_frame']['small_x_tick_interval']
            _ay = _tmp_conf['map_frame']['big_y_tick_interval']
            _fy = _tmp_conf['map_frame']['small_y_tick_interval']

        # REGION
        self.map_region = [minLon, maxLon, minLat, maxLat]

        # PROJECTION
        _projection = "%s%.3f/%.3f/%.3fc" % (
                        _tmp_conf['map_projection'].upper(),
                        np.float(meanLon),
                        np.float(meanLat),
                        np.float(_tmp_conf['fig_scale']))
        if _tmp_conf['map_projection'].upper() == "G":
            _projection += "+a30+t45+v60/60+w0+z250"
        self.map_projection = _projection

        # FRAME
        self.map_frame = []
        self.map_frame.append("xa%.1ff%.1f" % (_ax, _fx))
        self.map_frame.append("ya%.1ff%.1f" % (_ay, _fy))
        if _tmp_conf['map_frame']["show_axis"]:
            self.map_frame.append(_tmp_conf['map_frame']["show_axis"])

    def _setup_grid(self, data):
        """ For the moment simply associate the grid file """
        self.grid = data
        logger.debug("Correctly loaded GRID!")

    def _pd_select_columns(self, evid_prefix="csv_"):
        """ Select columns from pandas DataFrame and reorder them
            for the class API
        """
        _colnames = tuple(self.df.columns)
        _mandatory = set(("ID", "LON", "LAT", "DEP", "MAG"))
        # --- Search and change COLUMNS-NAME
        for cc in _colnames:
            if cc.lower().strip() in ("id", "event_id", "eqid", "eq_id", "#"):
                self.df.rename(columns={cc: "ID"}, errors="raise", inplace=True)
            # Origin Time
            elif cc.lower().strip() in ("ot", "origin_time", "utc_datetime", "utc"):
                self.df.rename(columns={cc: "OT"}, errors="raise", inplace=True)
            # Longitude
            elif cc.lower().strip() in ("lon", "longitude", "ev_longitude"):
                self.df.rename(columns={cc: "LON"}, errors="raise", inplace=True)
            # Latitude
            elif cc.lower().strip() in ("lat", "latitude", "ev_latitude"):
                self.df.rename(columns={cc: "LAT"}, errors="raise", inplace=True)
            # Depth
            elif cc.lower().strip() in ("dep", "depth", "ev_depth"):
                self.df.rename(columns={cc: "DEP"}, errors="raise", inplace=True)
            # Magnitude
            elif cc.lower().strip() in ("mag", "magnitude", "ev_magnitude"):
                self.df.rename(columns={cc: "MAG"}, errors="raise", inplace=True)
            # Magnitude Type
            elif cc.lower().strip() in ("magtype", "mag_type", "magnitude_type"):
                self.df.rename(columns={cc: "MAGTYPE"}, errors="raise", inplace=True)
            else:
                continue

        # --- Extract
        _new_colnames = set(self.df.columns)
        import pdb; pdb.set_trace()
        # fai il set difference --> se NON VUOTO, printa fuori i MANDATORI con ERR
        # Se difference vuoto, procedi col QUERY:
        self.df = self.df[["ID", "OT", "LON", "LAT", "DEP", "MAG", "MAGTYPE"]]

    def _setup_database(self, data):
        """ Switch among loading data ...
        """
        if isinstance(data, str):
            _data_path = Path(data)
            if _data_path.is_file() and _data_path.exists():
                _data_frame = pd.read_csv(_data_path)
                self._pandas2data(_data_frame)
            else:
                raise STE.FilesNotExisting("%s file do not exist!" % data)

        elif isinstance(data, pd.DataFrame):
            # pandas Dataframe --> simply append
            self._pandas2data(data)
            logger.debug("Recognized Pandas DataFrame data type ... loading")

        elif isinstance(data, obspy.Catalog):
            logger.debug("Recognized ObsPy Catalog data type ... loading")
            self._obspy2data(data)

        else:
            logger.error("Data type: %s  not yet supported! Contact maintaner" %
                         type(data))
        #
        logger.debug("Correctly loaded DATABASE!")

    def _pandas2data(self, data_frame):
        self.df = data_frame.copy()
        self._pd_select_columns()

    def _obspy2data(self, catalog, evid_prefix="opcat_"):
        """ If available, it will select the event's preferred solutions
        """
        evidlist = []
        utclist = []
        lonlist = []
        latlist = []
        deplist = []
        maglist = []
        magtype = []
        #
        for _xx, _ev in enumerate(catalog):
            if len(_ev.origins) != 0 and len(_ev.magnitudes) != 0:
                evidx = evid_prefix+str(_xx+1)
                # --- Select preferred solutions
                evor = _ev.preferred_origin()
                if not evor:
                    evor = _ev.origins[0]
                evmg = _ev.preferred_magnitude()
                if not evmg:
                    evmg = _ev.magnitudes[0]
                #
                evidlist.append(evidx)
                utclist.append(evor.time.datetime)
                lonlist.append(evor.longitude)
                latlist.append(evor.latitude)
                deplist.append(evor.depth*KM)
                maglist.append(evmg.mag)
                magtype.append(evmg.magnitude_type)
            else:
                logger.warning("(Event #%d) Have no origins OR magnitudes")

        # --- Create DATAFRAME
        self.df = pd.DataFrame(
                    {"ID": evidlist, "OT": utclist,
                     "LON": lonlist, "LAT": latlist, "DEP": deplist,
                     "MAG": maglist, "MAGTYPE": magtype}
                  )

    # === Setter
    def set_database(self, data):
        logger.info("Setting database file ...")
        self._setup_database(data)

    def set_gridfile(self, data=None):
        """ Data must be a file-path to a *grd file """
        if not data or data.lower() in ['default', 'd', 'def', 'global']:
            # Loading default relief file
            logger.info("Setting grid file ... DEFAULT")
            _tmp_grid = pygmt.datasets.load_earth_relief(resolution="10m")  # full globe
            self._setup_grid(_tmp_grid)
        else:
            logger.info("Setting grid file ... %s" % data)
            self._setup_grid(data)

    def set_configfile(self, config_file):
        logger.info("Configuring class with: %s" % config_file)
        self._setup_class_plot_map()
        self._setup_class_plot_sect()

    # === Getter
    def get_database(self):
        return self.df

    def get_gridfile(self):
        return self.grid

    # === Plotting
    def plot_map(self, map_config_file=None, show=True, store_name=None):
        """ Create Map using PyGMT library """

        if self.df is None or self.df.empty or not isinstance(self.df, pd.DataFrame):
            raise STE.MissingAttribute(
                "Missing DATA-FRAME object. Run `set_database` method first")

        if map_config_file:
            logger.info("Using configuration file:  %s" % map_config_file)
            self._setup_class_plot_map(map_config_file)

        # ======================================== FigComposition
        logger.info("Creating map for %d events ..." % self.df.shape[0])
        fig = pygmt.Figure()

        fig.basemap(region=self.map_region,
                    projection=self.map_projection,
                    frame=self.map_frame)

        pygmt.makecpt(cmap="lightgray", series=[200, 4000, 10])

        # # self.grid = pygmt.datasets.load_earth_relief(resolution="10m", region=self.map_region)
        # import pdb; pdb.set_trace()
        if self.grid is not None:
            _tmp_grid = pygmt.grdcut(self.grid, region=self.map_region)
            logger.debug("Plotting class grid-file")
            fig.grdimage(grid=_tmp_grid, shading=True, cmap="globe")  #cmap="lightgray")
            fig.coast(water="skyblue", shorelines=True, resolution='h')
        else:
            fig.coast(water="skyblue", land="gray",
                      shorelines=True, resolution='h')

        # ======================================== MainPlot
        fig.plot(x=self.df["LON"], y=self.df["LAT"], style="c0.3c",
                 color="white", pen="0.1,black")

        if show:
            fig.show(method="external")

        if isinstance(store_name, str):
            # remember to use extension "*.png - *.pdf"
            logger.info("Storing figure: %s" % store_name)
            fig.savefig(store_name)
        #
        return fig

    def plot_section():
        pass


# ================================================================
# ================================================  General TIPS
# dgrid = pygmt.grdgradient(grid=grid, radiance=[0, 0]) # custom radiance

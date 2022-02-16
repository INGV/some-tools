"""
This module is a wrapper for ADAPT-API results into GMT.
Thanks to PyGMT
"""

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
import some_tools.io as SIO


logger = logging.getLogger(__name__)

KM = 0.001
MT = 1000
DEFAULTSCONFIGFILE = (str(PurePath(ST.__file__).parent) +
                      "/config/pygmt_defaults.yml")

# ====================================================================
# ============================================  Module's Import Setup

# Set global PyGMT config
DEFAULTS = SIO._get_conf(DEFAULTSCONFIGFILE, check_version=True)
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

# =================================================================
# =================================================================
# ==============================================  MAP


class SomeMap(object):
    """ Base class for geographical plots based on several databases

    Supported database:
        - pandas.DataFrame
        - geopandas.DataFrame
        - obspy.Catalog

    Class Attributes:
        df (pandas.DataFrame)
        grid ()

    """
    def __init__(self, database, config_file=None, grid_data=None):
        self.df = None
        self.grid = None
        self.config_dict = None
        #
        self.region = None
        self.projection = None
        self.frame = None

        # # --------------- Init
        _loader = SIO.Loader(database)
        self.df = _loader.get_database()
        if grid_data:
            self._load_grid(grid_data)

        # MB: the following line will upload the default if not provided
        self._import_config_file(config_file=config_file)  # config_dict
        self._define_plot_parameters()

    def _load_grid(self, data=None):
        """ Data must be a file-path to a *grd file """
        if not data or data.lower() in ['default', 'd', 'def', 'global']:
            # Loading default relief file
            logger.info("Setting grid file ... DEFAULT")
            _tmp_grid = pygmt.datasets.load_earth_relief(resolution="10m")  # full globe
            self.grid = _tmp_grid
        else:
            logger.info("Setting grid file ... %s" % data)
            self.grid = data

    def _select_config_file(self, config_file=None):
        """ Simply return the configuration file DICT (after yalm read)
        """
        if config_file:
            _tmp_conf_dict = SIO._get_conf(config_file, check_version=True)
        else:
            _tmp_conf_dict = DEFAULTS
        return _tmp_conf_dict

    def _import_config_file(self, config_file=None):
        """ Defines MAP plot region, proj and frame from config file
        """
        _conf_dict = self._select_config_file(config_file)
        try:
            self.config_dict = _conf_dict["map_config"]
        except KeyError:
            raise STE.BadConfigurationFile(
              "Missing `map_config` key in config file!")

    def _define_plot_parameters(self):
        """ This method will return the values for REGION, PROJ, and
            FRAME. The format key is the following

            CONFIG_DICT:
                  auto_scale: True              # automatically determine the region interval and projection Lon/Lat
                  plot_region: [1, 21, 41, 51]  # Xmin, Xmax, Ymin, Ymax
                  expand_map_x: 1.0
                  expand_map_y: 1.0
                  plot_projection: m     # (M) mercator / (G) perspective / (L) Lambert
                  auto_frame: False
                  plot_frame:
                    big_x_tick_interval: 1.0
                    small_x_tick_interval: 0.5
                    big_y_tick_interval: 0.5
                    small_y_tick_interval: 0.25
                    annotate_axis: False  # combination of W E S N / False --> no ax-label shown
                  fig_scale: 12 # centimeter

        """
        if self.df is None or self.df.empty or not isinstance(self.df, pd.DataFrame):
            logger.warning("Missing database! `auto_scale` and `auto_frame` "
                           "options will be ignored!")

        if self.config_dict is None:
            raise STE.MissingAttribute(
                "Missing config-dict! "
                "Use set_config_file with a valid *.yml first!")

        logger.info("Setting-up the GMT section plot parameters! "
                    "(overriding if already present)")

        # ------- REGION
        Xmin = np.float(self.config_dict['plot_region'][0])
        Xmax = np.float(self.config_dict['plot_region'][1])
        Ymin = np.float(self.config_dict['plot_region'][2])
        Ymax = np.float(self.config_dict['plot_region'][3])
        #
        Xmean = Xmin + ((Xmax - Xmin)/2)
        Ymean = Ymin + ((Ymax - Ymin)/2)

        # REGION
        _region = [Xmin, Xmax, Ymin, Ymax]

        # PROJECTION
        _projection = "%s%.3f/%.3f/%.3fc" % (
                        self.config_dict['plot_projection'].upper(),
                        np.float(Xmean),
                        np.float(Ymean),
                        np.float(self.config_dict['fig_scale']))
        if self.config_dict['plot_projection'].upper() == "G":
            _projection += "+a30+t45+v60/60+w0+z250"

        # FRAME
        _ax = self.config_dict['plot_frame']['big_x_tick_interval']
        _fx = self.config_dict['plot_frame']['small_x_tick_interval']
        _ay = self.config_dict['plot_frame']['big_y_tick_interval']
        _fy = self.config_dict['plot_frame']['small_y_tick_interval']
        #
        _frame = []
        _frame.append("xa%.2ff%.2f" % (_ax, _fx))
        _frame.append("ya%.2ff%.2f" % (_ay, _fy))
        if self.config_dict['plot_frame']["annotate_axis"]:
            _frame.append(self.config_dict['plot_frame']["annotate_axis"])
        else:
            _frame.append("snwe")

        # ---------- Auto-Scale
        if self.config_dict['auto_scale']:
            _region = self._auto_scale_plot()
        if self.config_dict['auto_frame']:
            _frame = self._auto_frame_plot()

        # ---------- Allocate everything
        self.region = _region
        self.projection = _projection
        self.frame = _frame

    def _auto_scale_plot(self):
        """ Readjust class plot interval.
        """
        logger.warning(
            "Auto scaling the class REGION-plot attributes!")

        Xmin = np.float(np.min(self.df["LON"]))
        Xmax = np.float(np.max(self.df["LON"]))
        Ymin = np.float(np.min(self.df["LAT"]))
        Ymax = np.float(np.max(self.df["LAT"]))
        #
        _scaled_region = [Xmin, Xmax, Ymin, Ymax]
        return _scaled_region

    def _auto_frame_plot(self):
        """ Readjust class plot interval --> FRAME
        """
        logger.warning(
            "Auto scaling the class FRAME-plot attributes!")

        Xmin = np.float(np.min(self.df["LON"]))
        Xmax = np.float(np.max(self.df["LON"]))
        Ymin = np.float(np.min(self.df["LAT"]))
        Ymax = np.float(np.max(self.df["LAT"]))

        _ax = np.abs(np.float((Xmax-Xmin)/5.0))
        _fx = np.abs(np.float((Xmax-Xmin)/30.0))
        _ay = np.abs(np.float((Ymax-Ymin)/5.0))
        _fy = np.abs(np.float((Ymax-Ymin)/30.0))

        #
        _frame = []
        _frame.append("xa%.2ff%.2f" % (_ax, _fx))
        _frame.append("ya%.2ff%.2f" % (_ay, _fy))
        if self.config_dict['plot_frame']["annotate_axis"]:
            _frame.append(self.config_dict['plot_frame']["annotate_axis"])
        else:
            _frame.append("snwe")
        #
        return _frame

    # ========================================================= Setter
    def set_database(self, data):
        logger.info("Setting database file ...")
        _loader = SIO.Loader(data)
        self.df = _loader.get_database()

    def set_gridfile(self, data=None):
        """ Data must be a file-path to a *grd file """
        self._load_grid(data)

    def set_config_file(self, config_file=None):
        """ Defines MAP plot region, proj and frame from config file
        """
        logger.info("Configuring MAP class with: %s" % config_file)
        self._import_config_file(config_file=config_file)
        self._define_plot_parameters()

    # ========================================================= Getter
    def get_database(self):
        return self.df

    def get_gridfile(self):
        return self.grid

    def get_plot_parameters(self):
        od = {}
        od['region'] = self.region
        od['projection'] = self.projection
        od['frame'] = self.frame
        return od

    # ========================================================= Plotting
    def plot_map(self, plot_config=None,
                 show=True, store_name=None,
                 in_fig=None, panel=None):
        """ Create Map using PyGMT library """

        if self.df is None or self.df.empty or not isinstance(self.df, pd.DataFrame):
            raise STE.MissingAttribute(
                "Missing DATA-FRAME object. Run `set_database` method first")

        if isinstance(plot_config, dict):
            _plt_conf_dict = plot_config
        elif isinstance(plot_config, str):
            # Load config file and extract ONLY map_plot key
            _plt_conf_dict = SIO._get_conf(
                                plot_config, check_version=True)['map_plot']
        else:
            _plt_conf_dict = DEFAULTS["map_plot"]

        # ======================================== FigComposition
        logger.info("Creating MAP for %d events ..." % self.df.shape[0])
        fig = pygmt.Figure()

        fig.basemap(region=self.region,
                    projection=self.projection,
                    frame=self.frame)

        pygmt.makecpt(cmap="lightgray", series=[200, 4000, 10])

        if self.grid is not None and _plt_conf_dict["show_grid"]:
            _tmp_grid = pygmt.grdcut(self.grid, region=self.region)
            logger.debug("Plotting class grid-file")
            fig.grdimage(grid=_tmp_grid, shading=True, cmap="lightgray")  #cmap="globe"
            fig.coast(water="skyblue", shorelines=True, resolution='h')
        else:
            fig.coast(water="skyblue", land="gray",
                      shorelines=True, resolution='h')

        # ======================================== MainPlot
        _plot_df = self.df.sort_values(["MAG"], ascending=[False])

        if _plt_conf_dict['scale_magnitude']:
            fig.plot(x=_plot_df["LON"],
                     y=_plot_df["LAT"],
                     # size=0.03 * _plot_df["MAG"],
                     size=_plt_conf_dict['event_size'] * (2 ** _plot_df["MAG"]),
                     style="cc",
                     color=_plt_conf_dict['event_color'],
                     pen="0.25p,black")
        else:
            fig.plot(x=_plot_df["LON"],
                     y=_plot_df["LAT"],
                     size=np.full(_plot_df["MAG"].shape[0],
                                  _plt_conf_dict['event_size']),
                     style="cc",
                     color=_plt_conf_dict['event_color'],
                     pen="0.25p,black")

        if show:
            fig.show(method="external")

        if isinstance(store_name, str):
            # remember to use extension "*.png - *.pdf"
            logger.info("Storing figure: %s" % store_name)
            fig.savefig(store_name)
        #
        return fig

# =============================================================
# ================================================   SECTION
# =============================================================


class SomeSection(object):
    """ Base class to plot depth-sections of a seismic dataset
        along a fixed dataset

    Supported database:
        - pandas.DataFrame
        - geopandas.DataFrame
        - obspy.Catalog

    Class Attributes:
        df (pandas.DataFrame)
        grid ()

    """
    def __init__(self, database, config_file=None, grid_data=None):
        # --------------- Class Attributes
        self.df = None
        self.df_work = None  # the onw with projected events
        self.grid = None
        self.config_dict = None
        #
        self.region = None
        self.frame = None
        self.projection = None  # To define the section scale
        #
        self.fig_width = None
        self.fig_height = None

        # --------------- Init
        _loader = SIO.Loader(database)
        self.df = _loader.get_database()
        self._import_config_file(config_file=config_file)  # config_dict
        if grid_data:
            self._load_grid(grid_data)

        # Auto-Set Up
        self._project_dataframe()   # initialize the `df_work`
        self._define_plot_parameters()

    def _select_config_file(self, config_file=None):
        """ Simply return the configuration file DICT (after yalm read)
        """
        if config_file:
            _tmp_conf_dict = SIO._get_conf(config_file, check_version=True)
        else:
            _tmp_conf_dict = DEFAULTS
        return _tmp_conf_dict

    def _import_config_file(self, config_file=None):
        """ Defines MAP plot region, proj and frame from config file
        """
        _conf_dict = self._select_config_file(config_file)
        # check MAP key config
        try:
            self.config_dict = _conf_dict["sect_config"]
        except KeyError:
            raise STE.BadConfigurationFile(
              "Missing `sect_config` key in config file!")

    def _load_grid(self, data=None):
        """ Data must be a file-path to a *grd file """
        if not data or data.lower() in ['default', 'd', 'def', 'global']:
            # Loading default relief file
            logger.info("Setting grid file ... DEFAULT")
            _tmp_grid = pygmt.datasets.load_earth_relief(resolution="10m")  # full globe
            self.grid = _tmp_grid
        else:
            logger.info("Setting grid file ... %s" % data)
            self.grid = data

    def _define_plot_parameters(self):
        """ This method will return the values for REGION, PROJ, and
            FRAME. The format key is the following

            CONFIG_DICT:
                sect_config:
                  auto_scale: True                # automatically determine the region interval and projection Lon/Lat
                  section_profile: [1, 46, 21, 46]  # Xmin, Xmax, Ymin, Ymax
                  event_project_dist: "all" # All or float --> it will project +/- dist events
                  section_depth: [0, 10]
                  auto_frame: False
                  plot_frame:
                    big_x_tick_interval: 1.0
                    small_x_tick_interval: 0.5
                    big_y_tick_interval: 0.5
                    small_y_tick_interval: 0.25
                    annotate_axis: False  # combination of W E S N / False --> no ax-label shown
                  fig_scale: 12 # centimeter

        """
        if self.df_work is None or self.df_work.empty or not isinstance(self.df_work, pd.DataFrame):
            raise STE.MissingAttribute("Missing working-database! "
                                       "Run `_project_dataframe` method first!")

        if self.config_dict is None:
            raise STE.MissingAttribute(
                "Missing config-dict! "
                "Use set_config_file with a valid *.yml first!")

        #
        logger.info("Setting-up the GMT section plot parameters! "
                    "(overriding if already present)")

        # ------- REGION
        origin_X, _, end_X, _ = self._convert_coord_xy()
        _region = [origin_X, end_X,
                   self.config_dict['section_depth'][0],
                   self.config_dict['section_depth'][1]]

        # ------- PROJECTION
        _projection = "X%.3f/-%.3f" % (
                        np.float(self.config_dict['fig_dimension'][0]),
                        np.float(self.config_dict['fig_dimension'][1]))

        # ------- FRAME
        _ax = self.config_dict['plot_frame']['big_x_tick_interval']
        _fx = self.config_dict['plot_frame']['small_x_tick_interval']
        _ay = self.config_dict['plot_frame']['big_y_tick_interval']
        _fy = self.config_dict['plot_frame']['small_y_tick_interval']
        #
        _frame = []
        _frame.append("xa%.2ff%.2f" % (_ax, _fx))
        if self.config_dict['plot_frame']["show_grid_lines"]:
            _frame.append("ya%.2ff%.2fg%.2f@50" % (_ay, _fy, _ay))
        else:
            _frame.append("ya%.2ff%.2f" % (_ay, _fy))
        if self.config_dict['plot_frame']["annotate_axis"]:
            _frame.append(self.config_dict['plot_frame']["annotate_axis"])
        else:
            _frame.append("snwe")

        # ---------- Auto-Scale
        if self.config_dict['auto_scale']:
            _region = self._auto_scale_plot(lower_depth_fix=True)
        if self.config_dict['auto_frame']:
            _frame = self._auto_frame_plot()

        # Allocate everything
        self.region = _region
        self.projection = _projection
        self.frame = _frame
        #
        self.fig_width = self.config_dict['fig_dimension'][0]
        self.fig_height = self.config_dict['fig_dimension'][1]

    # ========= Provate WORK

    def _convert_coord_xy(self):
        """ Convert profile origin and end to cartesian coordinates """
        try:
            _ = self.config_dict["section_profile"]
        except KeyError:
            raise STE.MissingAttribute("I need the class attribute profile "
                                       "[lon1, lat1, lon2, lat2] to work on!")
        #
        _tmp_df = pd.DataFrame([
             [self.config_dict["section_profile"][0],
              self.config_dict["section_profile"][1]],
             [self.config_dict["section_profile"][2],
              self.config_dict["section_profile"][3]]
            ],
            columns=["LON", "LAT"])
        #
        _conv_df = pygmt.project(
                        _tmp_df,
                        convention="pq",
                        center=[self.config_dict["section_profile"][0],
                                self.config_dict["section_profile"][1]],
                        endpoint=[self.config_dict["section_profile"][2],
                                  self.config_dict["section_profile"][3]],
                        unit=True,
                        )
        #
        OriginX, OriginY = _conv_df[0][0], _conv_df[1][0]
        EndX, EndY = _conv_df[0][1], _conv_df[1][1]
        #
        return OriginX, OriginY, EndX, EndY

    def _auto_scale_plot(self, lower_depth_fix=True):
        """ Readjust class plot interval.
        If lower_depth_fix = True --> minimum_depth plot is equal to 0
        """
        if self.df_work is None or self.df_work.empty or not isinstance(self.df_work, pd.DataFrame):
            raise STE.MissingAttribute("Missing working-database! "
                                       "Run `_project_dataframe` method first!")
        #
        logger.warning(
            "Auto scaling the class REGION-plot attributes!")
        Xmin, Xmax = np.min(self.df_work[0]), np.max(self.df_work[0])
        Ymin, Ymax = np.min(self.df_work[2]), np.max(self.df_work[2])

        if lower_depth_fix:
            Ymin = 0
        #
        _scaled_region = [Xmin, Xmax, Ymin, Ymax]
        return _scaled_region

    def _auto_frame_plot(self):
        """ Readjust class plot interval --> FRAME
        """
        if self.df_work is None or self.df_work.empty or not isinstance(self.df_work, pd.DataFrame):
            raise STE.MissingAttribute("Missing working-database! "
                                       "Run `_project_dataframe` method first!")
        #
        logger.warning(
            "Auto scaling the class FRAME-plot attributes!")

        _ax = (np.max(self.df_work[0]) - np.min(self.df_work[0]))/6.0
        _fx = (np.max(self.df_work[0]) - np.min(self.df_work[0]))/35.0
        _ay = (np.max(self.df_work[2]) - np.min(self.df_work[2]))/6.0
        _fy = (np.max(self.df_work[2]) - np.min(self.df_work[2]))/35.0
        #
        _frame = []
        _frame.append("xa%.2ff%.2f" % (_ax, _fx))
        if self.config_dict['plot_frame']["show_grid_lines"]:
            _frame.append("ya%.2ff%.2fg%.2f@50" % (_ay, _fy, _ay))
        else:
            _frame.append("ya%.2ff%.2f" % (_ay, _fy))
        if self.config_dict['plot_frame']["annotate_axis"]:
            _frame.append(self.config_dict['plot_frame']["annotate_axis"])
        else:
            _frame.append("snwe")
        #
        return _frame

    def _project_dataframe(self):
        """ Here we create the `self.df_work` dataframe.
            return frame and region interval, as the figure projection
            is X (cartesian coordinates).
            Here we use pygmt.proj

        # work_df COLS:       0                           1           2      3
        #              dist_along_profile(km)   perp.distanze(km)   depth   mag
        """

        try:
            _ = self.config_dict["section_profile"]
        except KeyError:
            raise STE.MissingAttribute("I need the class attribute profile "
                                       "[lon1, lat1, lon2, lat2] to work on!")

        try:
            _ = self.config_dict["events_project_dist"]
        except KeyError:
            raise STE.MissingAttribute("I need the maximum distance for "
                                       "projecting earthquakes!")

        # -- Select width
        if isinstance(self.config_dict["events_project_dist"], (int, float)):
            _width = [-self.config_dict["events_project_dist"],
                      self.config_dict["events_project_dist"]]
        else:
            # Global
            _width = [-99999, 99999]

        logger.info("Creating projected DataFrame")
        self.df_work = pygmt.project(
            self.df[["LON", "LAT", "DEP", "MAG"]],
            convention="pqz",
            center=self.config_dict["section_profile"][0:2],
            endpoint=self.config_dict["section_profile"][2:4],
            length="w",
            width=_width,
            unit=True)  # p. q will be in km and not in degree! (use False otherwise)

        # Sort by magnitude for plot clearance
        self.df_work = self.df_work.sort_values(3, ascending=[False])

    def _gridtrack_elevation(self, increment=1.0, convert_to_km=True):
        """ Create topography profile to be placed adove the section!
            Increment is in km
        """

        logger.info("Creating elevation profile")

        _profile = pygmt.project(
            data=None,
            center=self.config_dict["section_profile"][0:2],
            endpoint=self.config_dict["section_profile"][2:4],
            length="w",
            generate=str(increment),
            unit=True)

        _track = pygmt.grdtrack(
            points=_profile,
            grid=self.grid,
            newcolname="elevation")

        if convert_to_km:
            _track[['elevation']] = _track[['elevation']]*KM

        return _track[['p', 'elevation']]

    # ========================================================= Setter
    def set_database(self, data):
        logger.info("Setting database file ...")
        _loader = SIO.Loader(data)
        self.df = _loader.get_database()

    def set_gridfile(self, data=None):
        """ Data must be a file-path to a *grd file """
        self._load_grid(data)

    def set_config_file(self, config_file=None):
        """ Defines MAP plot region, proj and frame from config file
            Simply reset and set everything for plotting again
        """
        logger.info("Configuring SECTION class with: %s" % config_file)
        self._import_config_file(config_file=config_file)
        self._update()

    def update_configuration(self, **kwargs):
        """ Update configuration parameters and go on """
        for kk, vv in kwargs.items():
            self.config_dict[kk] = vv
        #
        self._update()

    def _update(self):
        """ Just redo the update of plotting parameters """
        logger.info("Updating plotting parameters")
        self._project_dataframe()
        self._define_plot_parameters()

    # ========================================================= Getter
    def get_database(self):
        return self.df

    def get_gridfile(self):
        return self.grid

    def get_condig_dict(self):
        return self.config_dict

    def get_plot_parameters(self):
        od = {}
        od['section_profile'] = self.config_dict['section_profile']
        od['region'] = self.region
        od['projection'] = self.projection
        od['frame'] = self.frame
        return od

    # ========================================================= Plot
    def plot_section(self, plot_config=None,
                     show=True, store_name=None,
                     in_fig=None, panel=None):
        """ Create profile, plot section!
            If `in_fig` specified, it will be used instead (subplots!)
            if  `panel`  specified, will be used for plotting the axis
        """

        # ----------------------------------------- Extract config file
        if isinstance(plot_config, dict):
            _plt_conf_dict = plot_config
        elif isinstance(plot_config, str):
            # Load config file and extract ONLY map_plot key
            _plt_conf_dict = SIO._get_conf(
                                plot_config, check_version=True)['sect_plot']
        else:
            _plt_conf_dict = DEFAULTS["sect_plot"]

        # ----------------------------------------- PyGMT
        logger.info("Creating SECT for %d events ..." % self.df_work.shape[0])

        if in_fig and isinstance(in_fig, pygmt.Figure):
            do_subplot = True
            _fig = in_fig
        else:
            do_subplot = False
            _fig = pygmt.Figure()

            # _fig.set_panel(panel)

        if do_subplot and panel:
            with _fig.set_panel(panel):
                _fig.basemap(region=self.region,
                             projection=self.projection,
                             frame=self.frame)

                if _plt_conf_dict['scale_magnitude']:
                    _fig.plot(x=self.df_work[0],
                              y=self.df_work[2],
                              # size=0.03 * _plot_df["MAG"],
                              size=_plt_conf_dict['event_size'] * (2 ** self.df_work[3]),
                              style="cc",
                              color=_plt_conf_dict['event_color'],
                              pen="0.25p,black",
                              )
                else:
                    _fig.plot(x=self.df_work[0],
                              y=self.df_work[2],
                              size=np.full(self.df_work[3].shape[0],
                                           _plt_conf_dict['event_size']),
                              style="cc",
                              color=_plt_conf_dict['event_color'],
                              pen="0.25p,black",
                              )

        else:
            _fig.basemap(region=self.region,
                         projection=self.projection,
                         frame=self.frame)

            if _plt_conf_dict['scale_magnitude']:
                _fig.plot(x=self.df_work[0],
                          y=self.df_work[2],
                          size=_plt_conf_dict['event_size'] * (2 ** self.df_work[3]),
                          style="cc",
                          color=_plt_conf_dict['event_color'],
                          pen="0.25p,black",
                          )
            else:
                _fig.plot(x=self.df_work[0],
                          y=self.df_work[2],
                          size=np.full(self.df_work[3].shape[0],
                                       _plt_conf_dict['event_size']),
                          style="cc",
                          color=_plt_conf_dict['event_color'],
                          pen="0.25p,black",
                          )

        if show:
            _fig.show(method="external")

        if isinstance(store_name, str):
            # remember to use extension "*.png - *.pdf"
            logger.info("Storing figure: %s" % store_name)
            _fig.savefig(store_name)
        #
        return _fig

# =============================================================
# ================================================   SECTION
# =============================================================


class SomeElevation(object):
    """ Base class to plot elevation profiles of a topography grid
        along a fixed dataset

    The input-loading relies on PyGMT internal API

    """
    def __init__(self, grid_path, config_file=None):
        # --------------- Class Attributes
        self.grid = None
        self.profile = None
        self.config_dict = None
        #
        self.region = None
        self.frame = None
        self.projection = None  # To define the section scale
        #
        self.fig_width = None
        self.fig_height = None

        # --------------- Init
        self._load_grid(grid_path)
        self._import_config_file(config_file=config_file)  # config_dict

        # --------------- Auto-Set Up
        self._define_plot_parameters()

    def _select_config_file(self, config_file=None):
        """ Simply return the configuration file DICT (after yalm read)
        """
        if config_file:
            _tmp_conf_dict = SIO._get_conf(config_file, check_version=True)
        else:
            _tmp_conf_dict = DEFAULTS
        return _tmp_conf_dict

    def _import_config_file(self, config_file=None):
        """ Defines MAP plot region, proj and frame from config file
        """
        _conf_dict = self._select_config_file(config_file)
        # check MAP key config
        try:
            self.config_dict = _conf_dict["elevation_config"]
        except KeyError:
            raise STE.BadConfigurationFile(
              "Missing `sect_config` key in config file!")

    def _load_grid(self, data=None):
        """ Data must be a file-path to a *grd file """
        if not data or data.lower() in ['default', 'd', 'def', 'global']:
            # Loading default relief file
            logger.info("Setting grid file ... DEFAULT")
            _tmp_grid = pygmt.datasets.load_earth_relief(resolution="10m")  # full globe
            self.grid = _tmp_grid
        else:
            logger.info("Setting grid file ... %s" % data)
            self.grid = data

    def _define_plot_parameters(self):
        """ This method will return the values for REGION, PROJ, and
            FRAME. The format key is the following

            CONFIG_DICT:
                sect_config:
                  auto_scale: True                # automatically determine the region interval and projection Lon/Lat
                  section_profile: [1, 46, 21, 46]  # Xmin, Xmax, Ymin, Ymax
                  event_project_dist: "all" # All or float --> it will project +/- dist events
                  section_depth: [0, 10]
                  auto_frame: False
                  plot_frame:
                    big_x_tick_interval: 1.0
                    small_x_tick_interval: 0.5
                    big_y_tick_interval: 0.5
                    small_y_tick_interval: 0.25
                    annotate_axis: False  # combination of W E S N / False --> no ax-label shown
                  fig_scale: 12 # centimeter

            NEEDS the CONFIG FILE AND GRID

        """
        if not self.grid:
            raise STE.MissingAttribute("Missing working-database! "
                                       "Run `_project_dataframe` method first!")

        if self.config_dict is None:
            raise STE.MissingAttribute(
                "Missing config-dict! "
                "Use set_config_file with a valid *.yml first!")

        #
        logger.info("Setting-up the GMT section plot parameters! "
                    "(overriding if already present)")

        # ------- Extract profile
        self.profile = self._gridtrack_elevation(
                        increment=np.float(self.config_dict['sampling_dist']),
                        convert_to_km=self.config_dict['convert_to_km'])

        # ------- REGION
        origin_X, _, end_X, _ = self._convert_coord_xy()
        _region = [origin_X, end_X,
                   self.config_dict['section_elevation'][0],
                   self.config_dict['section_elevation'][1]]

        # ------- PROJECTION
        _projection = "X%f/%f" % (
                        np.float(self.config_dict['fig_dimension'][0]),
                        np.float(self.config_dict['fig_dimension'][1]))

        # ------- FRAME
        _ax = self.config_dict['plot_frame']['big_x_tick_interval']
        _fx = self.config_dict['plot_frame']['small_x_tick_interval']
        _ay = self.config_dict['plot_frame']['big_y_tick_interval']
        _fy = self.config_dict['plot_frame']['small_y_tick_interval']
        #
        _frame = []
        _frame.append("xa%.2ff%.2f" % (_ax, _fx))
        if self.config_dict['plot_frame']["show_grid_lines"]:
            _frame.append("ya%.2ff%.2fg%.2f@50" % (_ay, _fy, _ay))
        else:
            _frame.append("ya%.2ff%.2f" % (_ay, _fy))
        if self.config_dict['plot_frame']["annotate_axis"]:
            _frame.append(self.config_dict['plot_frame']["annotate_axis"])
        else:
            _frame.append("snwe")

        # ---------- Auto-Scale
        if self.config_dict['auto_scale']:
            try:
                _region = self._auto_scale_plot()
            except STE.MissingAttribute:
                logger.warning("Profile is EMPTY!")
        if self.config_dict['auto_frame']:
            try:
                _frame = self._auto_frame_plot()
            except STE.MissingAttribute:
                logger.warning("Profile is EMPTY!")

        # Allocate everything
        self.region = _region
        self.projection = _projection
        self.frame = _frame
        #
        self.fig_width = self.config_dict['fig_dimension'][0]
        self.fig_height = self.config_dict['fig_dimension'][1]

    # ========= Provate WORK

    def _convert_coord_xy(self):
        """ Convert profile origin and end to cartesian coordinates """
        try:
            _ = self.config_dict["section_profile"]
        except KeyError:
            raise STE.MissingAttribute("I need the class attribute profile "
                                       "[lon1, lat1, lon2, lat2] to work on!")
        #
        _tmp_df = pd.DataFrame([
             [self.config_dict["section_profile"][0],
              self.config_dict["section_profile"][1]],
             [self.config_dict["section_profile"][2],
              self.config_dict["section_profile"][3]]
            ],
            columns=["LON", "LAT"])
        #
        _conv_df = pygmt.project(
                        _tmp_df,
                        convention="pq",
                        center=[self.config_dict["section_profile"][0],
                                self.config_dict["section_profile"][1]],
                        endpoint=[self.config_dict["section_profile"][2],
                                  self.config_dict["section_profile"][3]],
                        unit=True,
                        )
        #
        OriginX, OriginY = _conv_df[0][0], _conv_df[1][0]
        EndX, EndY = _conv_df[0][1], _conv_df[1][1]
        #
        return OriginX, OriginY, EndX, EndY

    def _auto_scale_plot(self):
        """ Readjust class plot interval.
        """
        if self.profile is None or self.profile.empty or not isinstance(self.profile, pd.DataFrame):
            raise STE.MissingAttribute("Missing elevation profile! "
                                       "Run `_define_plot_parameters` method first!")
            return
        #
        logger.warning(
            "Auto scaling the class REGION-plot attributes!")

        Xmin, Xmax = np.min(self.profile['distance']), np.max(self.profile['distance'])
        Ymin, Ymax = np.min(self.profile['elevation']), np.max(self.profile['elevation'])

        _scaled_region = [Xmin, Xmax, Ymin, Ymax]
        return _scaled_region

    def _auto_frame_plot(self):
        """ Readjust class plot interval --> FRAME
        """
        if self.profile is None or self.profile.empty or not isinstance(self.profile, pd.DataFrame):
            raise STE.MissingAttribute("Missing working-database! "
                                       "Run `_project_dataframe` method first!")
        #
        logger.warning(
            "Auto scaling the class FRAME-plot attributes!")

        _ax = (np.max(self.profile['distance']) - np.min(self.profile['distance']))/6.0
        _fx = (np.max(self.profile['distance']) - np.min(self.profile['distance']))/35.0
        _ay = (np.max(self.profile['elevation']) - np.min(self.profile['elevation']))/6.0
        _fy = (np.max(self.profile['elevation']) - np.min(self.profile['elevation']))/35.0
        #
        _frame = []
        _frame.append("xa%.2ff%.2f" % (_ax, _fx))
        if self.config_dict['plot_frame']["show_grid_lines"]:
            _frame.append("ya%.2ff%.2fg%.2f@50" % (_ay, _fy, _ay))
        else:
            _frame.append("ya%.2ff%.2f" % (_ay, _fy))
        if self.config_dict['plot_frame']["annotate_axis"]:
            _frame.append(self.config_dict['plot_frame']["annotate_axis"])
        else:
            _frame.append("snwe")
        #
        return _frame

    def _gridtrack_elevation(self, increment=1.0, convert_to_km=True):
        """ Create topography profile to be placed adove the section!
            Increment is in km
        """

        logger.info("Creating elevation profile")

        _profile = pygmt.project(
            data=None,
            center=self.config_dict["section_profile"][0:2],
            endpoint=self.config_dict["section_profile"][2:4],
            length="w",
            generate=str(increment),
            unit=True)

        _track = pygmt.grdtrack(
            points=_profile,
            grid=self.grid,
            newcolname="elevation")

        if convert_to_km:
            _track[['elevation']] = _track[['elevation']]*KM

        _track.rename(columns={'p': 'distance'}, inplace=True)
        return _track[['distance', 'elevation']]

    # ========================================================= Setter
    def set_gridfile(self, data=None):
        """ Data must be a file-path to a *grd file """
        logger.info("Setting grid file ...")
        self._load_grid(data)

    def set_config_file(self, config_file=None):
        """ Defines MAP plot region, proj and frame from config file
            Simply reset and set everything for plotting again
        """
        logger.info("Configuring SECTION class with: %s" % config_file)
        self._import_config_file(config_file=config_file)
        self._update()

    def update_configuration(self, **kwargs):
        """ Update configuration parameters and go on """
        for kk, vv in kwargs.items():
            self.config_dict[kk] = vv
        #
        self._update()

    def _update(self):
        """ Just redo the update of plotting parameters """
        logger.info("Updating plotting parameters")
        self._define_plot_parameters()

    # ========================================================= Getter
    def get_database(self):
        return self.df

    def get_gridfile(self):
        return self.grid

    def get_condig_dict(self):
        return self.config_dict

    def get_plot_parameters(self):
        od = {}
        od['section_profile'] = self.config_dict['section_profile']
        od['region'] = self.region
        od['projection'] = self.projection
        od['frame'] = self.frame
        return od

    # ========================================================= Plot
    def plot_elevation_profile(self, plot_config=None,
                               show=True, store_name=None,
                               in_fig=None, panel=None):
        """ Create profile, plot section! """

        if not self.grid:
            raise STE.MissingAttribute("Missing class grid-file!")

        # ----------------------------------------- Extract config file
        if isinstance(plot_config, dict):
            _plt_conf_dict = plot_config
        elif isinstance(plot_config, str):
            # Load config file and extract ONLY map_plot key
            _plt_conf_dict = SIO._get_conf(
                                plot_config, check_version=True)['elevation_plot']
        else:
            _plt_conf_dict = DEFAULTS["elevation_plot"]

        # ----------------------------------------- PyGMT
        logger.info("Creating ELEVATION PROFILE ...")

        if in_fig and isinstance(in_fig, pygmt.Figure):
            do_subplot = True
            _fig = in_fig
        else:
            do_subplot = False
            _fig = pygmt.Figure()

        if do_subplot and panel:
            with _fig.set_panel(panel):
                _fig.plot(x=self.profile['distance'],
                          y=self.profile['elevation'],
                          pen="%s,%s" % (_plt_conf_dict['profile_width'],
                                         _plt_conf_dict['profile_color']),
                          #
                          region=self.region,
                          projection=self.projection,
                          frame=self.frame)
        else:
            _fig.plot(x=self.profile['distance'],
                      y=self.profile['elevation'],
                      pen="%s,%s" % (_plt_conf_dict['profile_width'],
                                     _plt_conf_dict['profile_color']),
                      #
                      region=self.region,
                      projection=self.projection,
                      frame=self.frame)

        if show:
            _fig.show(method="external")

        if isinstance(store_name, str):
            # remember to use extension "*.png - *.pdf"
            logger.info("Storing figure: %s" % store_name)
            _fig.savefig(store_name)
        #
        return _fig

# ================================================================
# ================================================  General TIPS
# dgrid = pygmt.grdgradient(grid=grid, radiance=[0, 0]) # custom radiance

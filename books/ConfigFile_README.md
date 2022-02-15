# SomeTools: plotting module config-file

Brief exaplanation of the **defaults** key and values meaning. The file must be with the `*.yml` extension (YAML).

## Global - GMT settings 
The global section is as follows:
```
some_tools_version: 0.0.4 

pygmt_config:
  MAP_FRAME_TYPE: plain
  FORMAT_GEO_MAP: ddd.xx
  MAP_GRID_CROSS_SIZE_PRIMARY: 0p
  MAP_GRID_CROSS_SIZE_SECONDARY: 0p
  PROJ_LENGTH_UNIT: c
  FONT_ANNOT_PRIMARY: 8p,0
  FONT_LABEL: 11p,1
```
The major parameters are:

- `some_tools_version`: package version, to avoid conflicts in keys naming etc ..
- `MAP_FRAME_TYPE`: defines the map and plots contours.
- `FORMAT_GEO_MAP`: defines the lon/lat tick labelling. This means that mast be in decimal degree format. If caps F is appended, also the component capital letter will be shown.
All the remaining and additional (why not?!) default parameters that can be changed  parameters can be found [here](https://docs.generic-mapping-tools.org/latest/gmt.conf.html.)

## SomeMap - settings 
```
# =============== Map related
map_config:
  auto_scale: True              
  plot_region: [1, 21, 41, 51]
  plot_projection: "m"
  auto_frame: True
  plot_frame:
    big_x_tick_interval: 1.0
    small_x_tick_interval: 0.5
    big_y_tick_interval: 0.5
    small_y_tick_interval: 0.25
    annotate_axis: False 
  fig_scale: 12

map_plot:
  event_color: "lightgray@20"
  event_size: 0.05
  scale_magnitude: True
  show_grid: True
```
__map_config__ is in charge of setting the region, frame, and projection of the maps. In particular:

- `plot_region` (list, tuple): is a list containing minimum/maximum longitude, and minimum/maximum latitude.
- `plot_projection` (str): is a letter describing which projection to adopt. At the moment (M)ercator, (G)perspective are supported. The projection lon0 and lat0 are determined by the mean longitude and mean latitude specified in the `plot_region` field.
- `auto_scale` (bool): if `True` it automatically determines the region interval extrema. It also affect the lon0 and lat0 values for plotting.
- `annotate_axis` (str): combination of W E S N to show both axis and label. If left empty or set to `False`, no axis label shown.
- `auto_frame` (bool): if `True` it automatically determines the frame's tick interval for both major and minor axis. If `False`, the values defined in the `plot_frame` section will be used. 
- `plot_frame[annotate_axis]` (str): combination of W E S N to show both axis and label. If left empty or set to `False`, no axis label shown.
- `fig_scale` (int, float): it reflects the figure width. It also directly determine the figure height depending on the projection used.

__map_plot__ is in charge of the plot details

- `scale_magnitude` (bool): if `True` it will plot the events circle size scaled by magnitude intensity
- `event_size` (int, float): represent a mag 1 eventsize if `scale_magnitude==True`. Fixed size otherwise.
- `show_grid` (bool): if set to `True`, the elevation grid-file (if present) will be loaded and shown.


## SomeSection - settings 
```
sect_config:
  auto_scale: True
  section_profile: [1, 46, 21, 46]  # lon1, lat1, lon2, lat2
  events_project_dist: "all" # All or float --> it will project +/- dist events
  section_depth: [0, 10]
  auto_frame: True
  plot_frame:
    big_x_tick_interval: 1.0
    small_x_tick_interval: 0.5
    big_y_tick_interval: 1.0
    small_y_tick_interval: 0.5
    show_grid_lines: True  # will plot major tick grid lines on Y-axis major-tick
    annotate_axis: "W"  # combination of W E S N / False --> no ax-label shown
  fig_dimension: [12, 4]  # x/y centimeter

sect_plot:
  event_color: "lightgray@20"
  event_size: 0.05       # float: represent a mag 1 eventsize if scale_magnitude==True
  scale_magnitude: True  # bool: if True plot scaled event
  plot_elevation: True
```

__sect_config__ is in charge of setting the region, frame, and projection of the depth-section. In particular:

__sect_plot__ is in charge of the plot details


```
# =============== Elevation related
elevation_config:
  auto_scale: True
  section_profile: [1, 46, 21, 46]  # lon1, lat1, lon2, lat2
  section_elevation: [-1, 6]  # km
  sampling_dist: 1  #km
  convert_to_km: True
  auto_frame: True
  plot_frame:
    big_x_tick_interval: 1.0
    small_x_tick_interval: 0.5
    big_y_tick_interval: 1.0
    small_y_tick_interval: 0.5
    show_grid_lines: True  # will plot major tick grid lines on Y-axis major-tick
    annotate_axis: "W"  # combination of W E S N / False --> no ax-label shown
  fig_dimension: [12, 1.5]  # x/y centimeter

elevation_plot:
  profile_width: "1p"
  profile_color: "black"
```
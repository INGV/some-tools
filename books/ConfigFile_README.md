# SomeTools: plotting module config-file

Brief exaplanation of the **defaults** key and values meaning. The file should be named with the `*.yml` extension (YAML), but it can be read with any (or no) etension. The most important thing is that all keys must be listed and with the correct formatting.

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
- `auto_frame` (bool): if `True` it automatically determines the frame's tick interval for both major and minor axis. If `False`, the values defined in the `plot_frame` section will be used. 
- `plot_frame[annotate_axis]` (str): combination of W E S N to show both axis and label. If left empty or set to `False`, no axis label shown.
- `fig_scale` (int, float): it reflects the figure width. It also directly determine the figure height depending on the projection used.

__map_plot__ is in charge of the plot details

- `scale_magnitude` (bool): if `True` it will plot the events circle size scaled by magnitude intensity.
- `event_size` (int, float): represent a mag 1 eventsize if `scale_magnitude==True`. Fixed size otherwise.
- `show_grid` (bool): if set to `True`, the elevation grid-file (if present) will be loaded and shown.
- `event_color` (bool): defines the color and transparency (with the `@`) of the events.


## SomeSection - settings 
```
sect_config:
  auto_scale: True
  section_profile: [1, 46, 21, 46]
  events_project_dist: "all" # All or float --> it will project +/- dist events
  section_depth: [0, 10]
  auto_frame: True
  plot_frame:
    big_x_tick_interval: 1.0
    small_x_tick_interval: 0.5
    big_y_tick_interval: 1.0
    small_y_tick_interval: 0.5
    show_grid_lines: True  # 
    annotate_axis: "W"  # combination of W E S N / False --> no ax-label shown
  fig_dimension: [12, 4]  # x/y centimeter

sect_plot:
  event_color: "lightgray@20"
  event_size: 0.05
  scale_magnitude: True  
```

__sect_config__ is in charge of setting the region, frame, and projection of the depth-section. In particular:

- `section_profile` (list, tuple): contains the lon/lat of the starting point and the lon/lat of the end-point.
- `events_project_dist` (int, float): correspond to +/- distance range of event projection. Any other types (i.e. str or bool) will result in projecting **all** the database on the profile. 
- `auto_scale` (bool): if `True` it automatically fit the plotting region interval extrema based on the event catalog distribution. It will not affect the projection meridian/parallel as it is a cartesian plot.
- `auto_frame` (bool): if `True` it automatically determines the frame's tick interval for both major and minor axis. If `False`, the values defined in the `plot_frame` section will be used. 
- `annotate_axis` (str): combination of W E S N to show both axis and label. If left empty or set to `False`, no axis label shown.
- `fig_dimension` (list, tuple): it reflects the figure width and height (determined by the `PROJ_LENGTH_UNIT` parameter). It also directly determine the figure height depending on the projection used.
- `show_grid_lines` (bool): if `True`, will plot major tick grid lines on Y-axis.

__sect_plot__ is in charge of the plot details:

- `scale_magnitude` (bool): if `True` it will plot the events circle size scaled by magnitude intensity
- `event_size` (int, float): represent a mag 1 eventsize if `scale_magnitude==True`. Fixed size otherwise.
- `event_color` (bool): defines the color and transparency (with the `@`) of the events.

## SomeElevation - settings 
```
elevation_config:
  auto_scale: True
  section_profile: [1, 46, 21, 46]
  section_elevation: [-1, 6]  # km
  sampling_dist: 1  #km
  convert_to_km: True
  auto_frame: True
  plot_frame:
    big_x_tick_interval: 1.0
    small_x_tick_interval: 0.5
    big_y_tick_interval: 1.0
    small_y_tick_interval: 0.5
    show_grid_lines: True
    annotate_axis: "W"
  fig_dimension: [12, 1.5]  # x/y centimeter

elevation_plot:
  profile_width: "1p"
  profile_color: "black@0"
```
__elevation_config__ is in charge of setting the region, frame, and projection of the depth-section. In particular:

- `section_profile` (list, tuple): contains the lon/lat of the starting point and the lon/lat of the end-point.
- `section_elevation` (list, tuple): elevation plot range (minimum, maximum). 
- `auto_scale` (bool): if `True` it automatically fit the plotting region interval extrema based on the event catalog distribution. It will not affect the projection meridian/parallel as it is a cartesian plot.
- `auto_frame` (bool): if `True` it automatically determines the frame's tick interval for both major and minor axis. If `False`, the values defined in the `plot_frame` section will be used.
- `show_grid_lines` (bool): if `True`, will plot major tick grid lines on Y-axis.
- `fig_dimension` (list, tuple): it reflects the figure width and height (determined by the `PROJ_LENGTH_UNIT` parameter). It also directly determine the figure height depending on the projection used.

__elevation_plot__ is in charge of the plot details:
 
- `profile_width` (str): pen width
- `profile_color` (str): pen color



## General advices:

- Use the `auto_scale` and `auto_frame` options just as a preliminary investigation handle. They will in fact tells you already the range of your dataset and the order of your labels. Once discovered, though, to achieve nicer plot one should explicit the plot parameter directly (i.e. when combining subplots).

- Beware of the indentation (i.e. `plot_frame` parameters), as they represent a separate, nested group.
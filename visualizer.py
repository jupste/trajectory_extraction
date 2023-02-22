import sys

sys.path.append('../data_analysis/')
from st_visions.st_visualizer import st_visualizer
import geopandas as gpd
import pandas as pd
from shapely import geometry
from bokeh.plotting import figure, output_file, save
from bokeh.tile_providers import get_provider, Vendors
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.models import HoverTool, CustomJS
from bokeh.layouts import row


def make_polygons(data):
    '''
    Create convex hull polygons from points in clusters

    Parameters
    ----------
    clusters : DataFrame
        point dataframe with cluster membership
    Returns
    clusters : DataFrame
        dataframe with shapely polygons covering the area of clusters
    '''
    clusters = data.copy()
    clusters.sort_values(by=['cluster'], ascending=[True], inplace=True)
    clusters.reset_index(drop=True, inplace=True)
    clusters['geometry'] = [geometry.Point(xy) for xy in zip(clusters['lon'], clusters['lat'])]
    poly_clusters = gpd.GeoDataFrame()
    gb = clusters.groupby('cluster')
    for y in gb.groups:
        df0 = gb.get_group(y).copy()
        point_collection = geometry.MultiPoint(list(df0['geometry']))
        convex_hull_polygon = point_collection.convex_hull
        poly_clusters = pd.concat(
            [poly_clusters, pd.DataFrame(data={'cluster_id': [y], 'geometry': [convex_hull_polygon]})])
    poly_clusters.reset_index(drop=True, inplace=True)
    poly_clusters.crs = 'epsg:4326'
    poly_clusters['size'] = poly_clusters.cluster_id.map(clusters.cluster.value_counts())
    return gpd.GeoDataFrame(poly_clusters, crs='epsg:4326')
def visualize_points(points, filename="data_points.html"):
    st_viz = st_visualizer(allow_complex_geometries=False)
    tools = None
    st_viz.set_data(points)
    basic_tools = "pan,box_zoom,wheel_zoom,save,reset,hover"
    extra_tools = f'{basic_tools},{",".join(tools)}' if tools is not None else basic_tools

    TOOLTIPS = [
        ("heading", "@heading"),
        ("index", "@index"),
        ("cluster", "@cluster")

    ]
    st_viz.create_canvas(title='Optimized DBCV score', sizing_mode='scale_width', height=500, width=500,
                         tools=extra_tools, tooltips=TOOLTIPS, )

    st_viz.figure.add_tile('ESRI_IMAGERY')
    st_viz.add_glyph(glyph_type='cross', color='red')
    # st_viz.show_figures(notebook=True, notebook_url='localhost:8888')
    output_file(filename=filename, title="DBSCAN clusters")
    save(st_viz.figure)


def visualize_polygons_and_points(poly, points=None, title='', filename='clusters.html'):
    st_viz = st_visualizer(allow_complex_geometries=False)
    tools = None
    basic_tools = ["pan", "box_zoom", "wheel_zoom", "save", "reset"]
    extra_tools = f'{basic_tools},{",".join(tools)}' if tools is not None else basic_tools
    st_viz.set_data(poly[poly.cluster_id != -1])
    #
    st_viz.create_canvas(title=title, sizing_mode='scale_width', height=200, width=300, tools=extra_tools)
    st_viz.figure.add_tile('ESRI_IMAGERY')

    st_viz.add_polygon(polygon_type='patches', line_color='blue', fill_color=None, line_width=2, alpha=1)
    # st_viz.figure.toolbar.active_scroll = st_viz.figure.select_one(bokeh_modelsWheelZoomTool)
    if points is not None:
        st_viz.set_data(points)
        st_viz.create_source()
        st_viz.add_glyph(glyph_type='circle', color='red', size=1)
        # st_viz.figure.add_tools(HoverTool(renderers = [st_viz.renderers[1]], tooltips=[("heading", "@heading")]))
    output_file(filename=filename, title=title)
    # save(row(st_viz.figure, s))
    save(st_viz.figure)


def visualize_polygons(poly, filename, poly2=None):
    st_viz = st_visualizer(allow_complex_geometries=False, limit=40000)
    tools = None
    basic_tools = ["pan", "box_zoom", "wheel_zoom", "save", "reset"]
    extra_tools = f'{basic_tools},{",".join(tools)}' if tools is not None else basic_tools
    TOOLTIPS = [
        ("index", "@cluster_id"),
        ("size", "@size")

    ]
    st_viz.set_data(poly[poly.cluster_id != -1])
    st_viz.create_canvas(title='', sizing_mode='scale_width', height=200, width=300, tools=extra_tools,
                         tooltips=TOOLTIPS)
    st_viz.figure.add_tile('ESRI_IMAGERY')
    st_viz.add_polygon(polygon_type='patches', line_color='red', fill_color='blue', line_width=5, alpha=1)

    # st_viz.figure.sizing_mode = 'scale_height'
    if poly2 is not None:
        st_viz.set_data(poly2[poly2.cluster_id != -1])
        st_viz.create_source()
        st_viz.add_polygon(polygon_type='patches', line_color='blue', line_width=2, fill_color='white')
    output_file(filename=filename, title="Bounding box")
    save(st_viz.figure)
# pyright: basic
import geopandas as gpd
from shapely import geometry

from src.processing import processors


def test_extract_green_points_namespaced_fids():
    """Regression: green and trees data ids must not collide.

    cityseer deduplicates by data_id globally, so if both green and trees
    polygons carried a bare RangeIndex (both index 0), a nearer tree point
    could mask a green polygon.  process_green namespaces fids per category
    (green_<fid> / trees_<fid>) via _extract_green_points.
    """
    crs = 3035
    # both frames carry RangeIndex 0 — un-namespaced fids would collide
    green_gdf = gpd.GeoDataFrame(geometry=[geometry.box(0, 0, 100, 100)], crs=crs)
    trees_gdf = gpd.GeoDataFrame(geometry=[geometry.box(200, 0, 300, 100)], crs=crs)
    points_gdf = processors._extract_green_points(green_gdf, trees_gdf)
    # both categories produced points
    green_fids = set(points_gdf.loc[points_gdf["cat"] == "green", "fid"])
    trees_fids = set(points_gdf.loc[points_gdf["cat"] == "trees", "fid"])
    assert green_fids == {"green_0"}
    assert trees_fids == {"trees_0"}
    # namespacing keeps the shared RangeIndex from colliding across categories
    assert not green_fids & trees_fids

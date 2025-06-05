import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString

# def draw(*objs):
f, ax = plt.subplots()
    # for obj in objs:
    #     obj.plot(ax=ax)
    # plt.show()
    # f.waitforbuttonpress()

square = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])])

points = gpd.GeoDataFrame(geometry=gpd.GeoSeries([
    square.centroid.iloc[0],
    Point(0.5, 1.5),
    Point(1.5, 0.5),
    Point(0.5, -0.5),
    Point(-0.5, 0.5),
]))

square.boundary.plot(ax=ax, linewidth=4, color="red")

def vor(points):
    v = gpd.GeoDataFrame(geometry=points.voronoi_polygons(), crs=points.crs)
    return gpd.sjoin(v, points, how='left', predicate='contains', rsuffix='point')
    # v = gpd.GeoDataFrame(geometry=points.voronoi_polygons(), crs=points.crs)
    # joined = gpd.sjoin(points, v, how="left", predicate="within")
    # return v.merge(joined[["tzid", "index_right"]], left_index=True, right_on="index_right").drop(columns=["index_right"])

    # return points.voronoi_polygons()
    
def polygons_contains_point(polys, point) -> Polygon:
    # get the point from points.iloc[0] or similar
    # NOT points.iloc[[0]]
    # this only checks for a single point
    return polys[polys.geometry.contains(point)].iloc[0]


def vadjacent(points, point: Point, v):
    # get point from points with points.iloc[idx].geometry
    # point = points.iloc[idx].geometry
    poly = polygons_contains_point(v, point)
    npoly = v[v.touches(poly.geometry)]
    return points.iloc[npoly.index_point]


    
points.plot(ax=ax)

def vnormals(points, v):
    return pd.concat([vnormal(points, n, v) for n, _ in points.iterrows()])


def vnormal(points, idx, v):
    point = points.iloc[idx]
    neighbors = vadjacent(points, point.geometry, v)
    return gpd.GeoSeries([
        LineString([point.geometry, neighbor.geometry])
        for n, neighbor in neighbors.iterrows()
    ])

# def vedge(

v = vor(points)
v.boundary.plot(ax=ax) #, linewidth=2, linestyle='dashed')
vnormals(points, v).plot(ax=ax, linewidth=0.5, linestyle='dotted')

if __name__ == '__main__':
    plt.show()
    # draw(square.boundary, points, v(points).boundary)

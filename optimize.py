import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from shapely.geometry import Polygon, Point, LineString

# def draw(*objs):
f, ax = plt.subplots()
    # for obj in objs:
    #     obj.plot(ax=ax)
    # plt.show()
    # f.waitforbuttonpress()

square = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)])])

# points = gpd.GeoDataFrame(geometry=gpd.GeoSeries([
#     square.centroid.iloc[0],
#     Point(0.5, 1.5),
#     Point(1.5, 0.5),
#     Point(0.5, -0.5),
#     Point(-0.5, 0.5),
# ]))


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
    return polys[polys.geometry.contains(point)].iloc[0].geometry

def vadjacent_polys(point: Point, v):
    # get point from points with points.iloc[idx].geometry
    # point = points.iloc[idx].geometry
    poly = polygons_contains_point(v, point)
    # https://stackoverflow.com/questions/28028910/how-to-deal-with-rounding-errors-in-shapely
    # npoly = v[v.touches(poly)]
    npoly = v[v.geometry.intersection(poly).geom_type == 'LineString']
    return npoly


def vadjacent(points, point: Point, v):
    # get point from points with points.iloc[idx].geometry
    # point = points.iloc[idx].geometry
    npoly = vadjacent_polys(point, v)
    return points.iloc[npoly.index_point]

def gdfgs(*poly):
    return gpd.GeoDataFrame(geometry=gpd.GeoSeries(poly))



def vnormals(points, v):
    return pd.concat([vnormal(points, point, v) for _, point in points.iterrows()], ignore_index=True)


def vnormal(points, point, v):
    # point HAS to be a part of points
    # point is what you get when you do points.iloc[n]
    neighbors = vadjacent(points, point.geometry, v)
    return gpd.GeoSeries([
        LineString([point.geometry, neighbor.geometry])
        for _, neighbor in neighbors.iterrows()
    ])

def edges_from_shape(s: Polygon):
    # pass iloc[0].geometry into this
    b = s.boundary.coords
    return gpd.GeoSeries([
        LineString(b[k:k+2])
        for k
        in range(len(b) - 1)
    ])


def get_intersection_with_circle(midpoint, direction_vector, radius):
    """
    Given a midpoint and a direction vector, return two intersection points
    of the line through the midpoint with a circle of given radius.
    """
    # Normalize the direction vector
    unit_vec = direction_vector / np.linalg.norm(direction_vector)

    # Scale to the radius
    offset = unit_vec * radius

    # Two points at ± offset from the midpoint
    p1 = midpoint + offset
    p2 = midpoint - offset

    return gpd.GeoSeries([Point(p1), Point(p2)])

def vedge(edge: LineString, d: int = 1):

    # turn it into a vector
    coords = np.array(edge.coords)
    start, end = coords

    # get the midpoint
    midpoint = (start + end) / 2

    vec = end - start
    vec = np.array([-vec[1], vec[0]]) # rotate 90 degrees

    # get the two points on the vector «d» units away from midpoint, circle intersection
    return get_intersection_with_circle(midpoint, vec, d)

edge = edges_from_shape(square.iloc[0].geometry).iloc[0]




axslider = f.add_axes([0.25, 0.05, 0.65, 0.03])

slider = Slider(
    ax=axslider,
    label='d',
    valmin=1,
    valmax=50,
    valinit=30,
)

def update(val):
    ax.clear()
    ax.set_xlim([-100, 200])
    ax.set_ylim([-100, 200])
    square.boundary.plot(ax=ax, linewidth=4, color="red")
    points = gpd.GeoDataFrame(
        geometry=pd.concat([vedge(edge, d=slider.val) for edge in edges_from_shape(square.iloc[0].geometry)], ignore_index=True)
    )

    points.plot(ax=ax)

    v = vor(points)
    v.boundary.plot(ax=ax) #, linewidth=2, linestyle='dashed')
    vnormals(points, v).plot(ax=ax, linewidth=0.5, linestyle='dotted')

    f.canvas.draw_idle()

slider.on_changed(update)


# a = points.iloc[[0]]


# poly = polygons_contains_point(v, a.iloc[0].geometry)
# gpd.GeoDataFrame(geometry=gpd.GeoSeries([poly])).plot(ax=ax)
# vadjacent_polys(a.iloc[0].geometry, v).plot(ax=ax, linewidth=3, color='yellow')


if __name__ == '__main__':
    plt.show()
    # draw(square.boundary, points, v(points).boundary)

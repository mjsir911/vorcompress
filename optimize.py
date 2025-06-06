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


if 1:
    bottomleft = (-30, -10)
    topright = (30, 30)
    square = gpd.GeoDataFrame.from_file('wasd.shp')
else:
    bottomleft = (-200, -200)
    topright = (200, 200)
    square = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (10, -80), (130, 0), (120, 70), (30, 100), (0, 100), (-30, 50), (0, 0)])])
    # triangle
    # square = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (10, -80), (-30, 50), (0, 0)])])
    # square = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])])
    # square = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (100, 0), (100, 80), (80, 100), (0, 100)])])

ax.set_xlim([bottomleft[0], topright[0]])
ax.set_ylim([bottomleft[1], topright[1]])
window = Polygon([bottomleft, (bottomleft[0], topright[1]), topright, (topright[0], bottomleft[1])])



# square = original_borders

# points = gpd.GeoDataFrame(geometry=gpd.GeoSeries([
#     square.centroid.iloc[0],
#     Point(0.5, 1.5),
#     Point(1.5, 0.5),
#     Point(0.5, -0.5),
#     Point(-0.5, 0.5),
# ]))


def vor(points):
    # v = gpd.GeoDataFrame(geometry=points.voronoi_polygons(), crs=points.crs)
    v = gpd.GeoDataFrame(geometry=points.voronoi_polygons(extend_to=window), crs=points.crs)
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


def my_intersects(a, b):
    # https://stackoverflow.com/questions/28028910/how-to-deal-with-rounding-errors-in-shapely
    # return a[a.touches(b)]
    # return a[a.geometry.intersection(b).geom_type == 'LineString']
    intersections = a.geometry.intersection(b)
    return a[~intersections.is_empty & (intersections.geom_type == 'LineString')]

def vadjacent_polys(point: Point, v):
    # get point from points with points.iloc[idx].geometry
    # point = points.iloc[idx].geometry
    poly = polygons_contains_point(v, point)
    npoly = my_intersects(v, poly)
    return npoly


def vadjacent(points, point: Point, v):
    # get point from points with points.iloc[idx].geometry
    # point = points.iloc[idx].geometry
    npoly = vadjacent_polys(point, v)
    return points.iloc[npoly.index_point]

def gdfgs(*poly):
    return gpd.GeoDataFrame(geometry=gpd.GeoSeries(poly))



def vnormals(points, v):
    return pd.concat([vnormal(points, point.geometry, v) for _, point in points.iterrows()], ignore_index=True)


def vnormal(points, point, v):
    # point HAS to be a part of points
    # point is what you get when you do po3nts.iloc[n].geometry
    neighbors = vadjacent(points, point, v)
    return gpd.GeoSeries([
        LineString([point, neighbor.geometry])
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
    # chatgpt kind of

    # turn it into a vector
    coords = np.array(edge.coords)
    start, end = coords

    # get the midpoint
    midpoint = (start + end) / 2

    vec = end - start
    vec = np.array([-vec[1], vec[0]]) # rotate 90 degrees

    # get the two points on the vector «d» units away from midpoint, circle intersection
    return get_intersection_with_circle(midpoint, vec, d)




square.boundary.plot(ax=ax, linewidth=4, color="red", linestyle='dashed')



def get_triangle_circumcenter(triangle: Polygon) -> Point:
    # return triangle.centroid
    # chatgpt
    # Ensure it's a triangle
    coords = list(triangle.exterior.coords)

    # A valid triangle has 4 coordinates (first and last are the same in shapely polygons)
    assert len(coords) == 4, "Polygon must be a triangle (3 sides)"

    # Extract the 3 unique points
    A, B, C = coords[0], coords[1], coords[2]

    # Unpack coordinates
    x1, y1 = A
    x2, y2 = B
    x3, y3 = C

    # Compute circumcenter using determinant-based formula
    D = 2 * (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
    assert D != 0, "Points must not be colinear"

    Ux = ((x1**2 + y1**2)*(y2 - y3) +
          (x2**2 + y2**2)*(y3 - y1) +
          (x3**2 + y3**2)*(y1 - y2)) / D

    Uy = ((x1**2 + y1**2)*(x3 - x2) +
          (x2**2 + y2**2)*(x1 - x3) +
          (x3**2 + y3**2)*(x2 - x1)) / D

    return Point(Ux, Uy)

# d.boundary.plot(ax=ax, color="green")

def mirror_point_about_perpendicular(point: Point, edge: LineString) -> Point:
    if len(edge.coords) != 2:
        raise ValueError("Edge must be a LineString with exactly 2 points.")

    # Convert to numpy arrays
    P = np.array(*point.coords)

    # 4. Determine which side P lies on by checking vector from projection to P
    projection = edge.interpolate(edge.project(point))  # closest point on line to P
    proj_coords = np.array([projection.x, projection.y])
    vec_proj_to_P = P - proj_coords

    return Point(P - (P - proj_coords) * 2)


def genpoints0():
    points = gpd.GeoDataFrame(geometry=square.centroid)
    return iteratepoints1(points)


def genpoints2():
    points = gdfgs(*[Point(coord) for coord in square.exterior.iloc[0].coords])

    d = points.delaunay_triangles()

    points = gdfgs(*[
        get_triangle_circumcenter(dp)
        for dp
        in points.delaunay_triangles()
    ])

    # points = points[points.geometry.within(square)]

    return iteratepoints1(points)
    # return filterpoints(iteratepoints1(points), square)

def genpoints3():
    points = gdfgs(*[Point(coord) for coord in square.exterior.iloc[0].coords])

    d = points.delaunay_triangles()
    # d.plot(color='green', ax=ax)

    points = gdfgs(*[
        dp.centroid
        # triangle_incenter(dp)
        for dp
        in points.delaunay_triangles()
    ])

    return iteratepoints1(points)

def genpoints4():
    points = gdfgs(*[Point(coord) for coord in square.exterior.iloc[0].coords])

    d = points.delaunay_triangles()
    # d.boundary.plot(color='green', ax=ax)

    points = gdfgs(*[
        triangle_incenter(dp)
        for dp
        in points.delaunay_triangles()
    ])

    points = iteratepoints1(points)
    # points = iteratepoints1(points)
    return points


def triangle_incenter(t):
    # Get coordinates of the triangle
    coords = list(t.exterior.coords)[:3]  # Only the first 3, skip closing point

    # Convert to NumPy array for easier math
    A, B, C = np.array(coords[0]), np.array(coords[1]), np.array(coords[2])

    # Calculate lengths of the sides opposite each vertex
    a = np.linalg.norm(B - C)  # length of side opposite A
    b = np.linalg.norm(C - A)  # length of side opposite B
    c = np.linalg.norm(A - B)  # length of side opposite C

    # Calculate incenter using weighted average of vertices
    incenter = (a * A + b * B + c * C) / (a + b + c)

    # Convert to shapely Point
    incenter_point = Point(incenter)
    return incenter_point


def filterpoints(points, shape):
    v = vor(points)
    print(points)
    toremove = []
    for n, point in enumerate(points.geometry):
        poly = polygons_contains_point(v, point)
        if not poly.intersects(shape).iloc[0].geometry:
            toremove.append(n)
    return points.drop(toremove)



# d.boundary.plot(ax=ax, color="green")


edges = edges_from_shape(square.iloc[0].geometry)


# points = gpd.GeoDataFrame(geometry=square.centroid)

# points = points[points.within(square.iloc[0].geometry)]


def genpoints1(d=10):
    edges = edges_from_shape(square.iloc[0].geometry)
    return gpd.GeoDataFrame(geometry=pd.concat([
        vedge(edge, d)
        for edge
        in edges
    ], ignore_index=True))


def iteratepoints1(points):
    v = vor(points)
    return pd.concat([
        points,
        *[gdfgs(*[
            mirror_point_about_perpendicular(p, edge)
            for edge
            in edges[edges.intersects(polygons_contains_point(v, p))]
        ])
        # for p in points[points.within(square.iloc[0].geometry)].geometry
        for p in points.geometry
        ]],
        ignore_index=True
    )


def optimizepoints(points, shape, v=None):
    if v is None:
        v = vor(points)
    droppable = []

    edges = edges_from_shape(square.iloc[0].geometry)

    for n, point in points.iterrows():
        poly = polygons_contains_point(v, point.geometry)
        if not (any(edges.intersects(poly)) or any(edges.touches(poly))):
            droppable.append(n)

    return points.drop(droppable)



def vdraw(points, v):
    points.plot(ax=ax)
    v.boundary.plot(ax=ax) #, linewidth=2, linestyle='dashed')
    # vnormals(points, v).plot(ax=ax, linewidth=0.5, linestyle='dotted')

points = genpoints4()
print(points)
# points = optimizepoints(points, square)
v = vor(points)
vdraw(points, v)





def update(val):
    ax.clear()
    ax.set_xlim([bottomleft[0], topright[0]])
    ax.set_ylim([bottomleft[1], topright[1]])
    square.boundary.plot(ax=ax, linewidth=4, color="red")

    edges = edges_from_shape(square.iloc[0].geometry)

    points = genpoints1(d=slider.val)

    v = vor(points)

    vdraw(points, v)

    f.canvas.draw_idle()

# axslider = f.add_axes([0.25, 0.05, 0.65, 0.03])
#
# slider = Slider(
#     ax=axslider,
#     label='d',
#     valmin=1,
#     valmax=50,
#     valinit=30,
# )
#
# slider.on_changed(update)

# a = points.iloc[[0]]


# poly = polygons_contains_point(v, a.iloc[0].geometry)
# gpd.GeoDataFrame(geometry=gpd.GeoSeries([poly])).plot(ax=ax)
# vadjacent_polys(a.iloc[0].geometry, v).plot(ax=ax, linewidth=3, color='yellow')


if __name__ == '__main__':
    plt.show()
    # draw(square.boundary, points, v(points).boundary)

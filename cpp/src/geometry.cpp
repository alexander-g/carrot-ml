#include <algorithm>
#include <cmath>
#include <limits>

#include "./geometry.hpp"




static inline double cross2d(const Point &a, const Point &b, const Point &c) {
    // cross product of AB x AC (z-component)
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
}

inline bool point_on_segment(const Point &p, const Point &a, const Point &b) {
    // Check collinearity and that p is between a and b (inclusive).
    const double eps = 1e-12;
    if (std::fabs(cross2d(a, b, p)) > eps) 
        return false;
    // Check bounding box
    const double minx = std::min(a[0], b[0]) - eps;
    const double maxx = std::max(a[0], b[0]) + eps;
    const double miny = std::min(a[1], b[1]) - eps;
    const double maxy = std::max(a[1], b[1]) + eps;
    return (p[0] >= minx && p[0] <= maxx && p[1] >= miny && p[1] <= maxy);
}



inline bool point_in_polygon(const Point& p, const Path& polygon) {
    // TODO: remove last point if duplicate ?

    const int n = polygon.size();
    if(n < 3)
        return false;

    for(int i = 0, j = n - 1; i < n; j = i++) {
        if(point_on_segment(p, polygon[j], polygon[i])) 
            return true;
    }

    // count crossings of a horizontal ray to the right
    bool inside = false;
    for(int i = 0, j = n - 1; i < n; j = i++) {
        const Point &vi = polygon[i];
        const Point &vj = polygon[j];

        // do the endpoints lie on opposite sides of the horizontal line y = p.y ?
        const bool different_sides = ((vi[1] > p[1]) != (vj[1] > p[1]));
        const bool intersect = 
            different_sides &&
            (p[0] < (vj[0] - vi[0]) * (p[1] - vi[1]) / (vj[1] - vi[1]) + vi[0]);

        if(intersect) 
            inside = !inside;
    }

    return inside;
}


std::vector<bool> points_in_polygon(const Points& points, const Path& polygon) {
    std::vector<bool> output;
    for(const Point& p: points) 
        output.push_back( point_in_polygon(p, polygon) );
    return output;
}

/** Euclidean distance */
double distance(const Point& p0, const Point& p1) {
    const double d0 = p0[0] - p1[0];
    const double d1 = p0[1] - p1[1];
    return sqrt(  d0*d0 + d1*d1  );
}

std::vector<double> points_to_point_distances(const Points& points, const Point& p) {
    std::vector<double> output;
    output.reserve(points.size());
    for(const Point& p_i: points)
        output.push_back( distance(p_i, p) );
    return output;
}

std::optional<double> closest_distance(const Points& points, const Point& p) {
    if(points.empty())
        return std::nullopt;

    const auto distances = points_to_point_distances(points, p);
    return *std::min_element(distances.begin(), distances.end());
}

std::optional<Point> average_points(const Points& points) {
    if(points.size() == 0)
        return std::nullopt;
    
    Point sum = {0,0};
    for(const Point& p: points){
        sum[0] += p[0];
        sum[1] += p[1];
    }
    const int n = points.size();
    return Point{sum[0] / n, sum[1] / n};
}


/** Rescale points from one image shape to another.
    Points are expected to be in XY format, shapes in HW. */
Points scale_points(
    const Points& points_xy, 
    const ImageShape& from_shape, 
    const ImageShape& to_shape
) {
    const std::pair<double, double> scale = {
        to_shape.first  / (double)from_shape.first,    // height
        to_shape.second / (double)from_shape.second    // width
    };
    Points output;
    for(const Point& point: points_xy)
        output.push_back({point[0] * scale.second, point[1] * scale.first});
    return output;
}


/** Rescale nested points from one image shape to another.
    Points are expected to be in XY format, shapes in HW. */
ListOfPoints scale_list_of_points(
    const ListOfPoints& points_xy,
    const ImageShape& from_shape, 
    const ImageShape& to_shape 
) {
    ListOfPoints output;
    for(const Points& path: points_xy)
        output.push_back( scale_points(path, from_shape, to_shape) );
    return output;
}


/** Convert pixel indices (y first, x second) to points (x first, y second),
    optionally centering on the pixel (+0.5). */
Points indices_to_points(const Indices2D& indices, bool center_pixel){
    const double offset = 0.5 * center_pixel;
    Points points;
    for(const Index2D& index: indices)
        // yx to xy
        points.push_back({ (double)index.j + offset, (double)index.i + offset });
    return points;
}



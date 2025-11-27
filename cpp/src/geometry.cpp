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


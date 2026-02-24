#pragma once

#include <array>
#include <optional>
#include <utility>
#include <vector>

#include "../wasm-morpho/src/morphology.hpp"


// x first, y second
typedef std::array<double, 2> Point;
typedef std::array<double, 2> Vector;
typedef std::vector<Point>    Points;
typedef std::vector<Point>    Path;
typedef std::vector<Path>     Paths;
typedef std::vector<Points>   ListOfPoints;
typedef std::pair<Path, Path> PathPair;
typedef std::vector<PathPair> PairedPaths;

// NOTE: height first, width second
typedef std::pair<int,int>    ImageShape;


struct Box {
    double x0; // left
    double y0; // top
    double x1; // right
    double y1; // bottom
};

/** Line coefficients */
typedef struct LineCoeffs {
    // x
    double a;
    // y
    double b;
    // offset
    double c;
} LineCoeffs;



struct AreaOfInterestRect {
    Point p0, p1, p2, p3;
};




/** Normalize x to unit length */
Vector normalize(const Vector& v);

/** Compute the coefficients of a line going through the points `p0` and `p1 */
LineCoeffs line_from_two_points(const Point& p0, const Point& p1);


/** Evaluate the equation ax + by + c, resulting in the signed distance of `p` 
    to the line. 
    NOTE: needs to stay in the header file, slow in the browser otherwise. */
inline double eval_implicit_equation(const LineCoeffs& coef, const Point& p) {
    //const Vector normcoef = normalize({coef.a, coef.b});
    //return p[0] * normcoef[0]  +  p[1] * normcoef[1]  +  coef.c;
    return p[0] * coef.a  +  p[1] * coef.b  +  coef.c;
}


/** Test if point p lies on the line segment a - b */
bool point_on_segment(const Point& p, const Point& a, const Point& b);

/** Test if points lie inside of a polygon */
std::vector<bool> points_in_polygon(const Points& points, const Path& polygon);

/** Euclidean distance */
double distance(const Point& p0, const Point& p1);

/** Compute distances from points `p` to all in `points` */
std::vector<double> points_to_point_distances(const Points& points, const Point& p);

/** Compute the distance from points `p` to the closest one in `points` */
std::optional<double> closest_distance(const Points& points, const Point& p);

/** Compute the average point */
std::optional<Point> average_points(const Points& points);

/** Rescale points from one image shape to another.
    Points are expected to be in XY format, shapes in HW. */
Points scale_points(
    const Points& points_xy, 
    const ImageShape& from_shape, 
    const ImageShape& to_shape
);

/** Rescale nested points from one image shape to another.
    Points are expected to be in XY format, shapes in HW. */
ListOfPoints scale_list_of_points(
    const ListOfPoints& points_xy,
    const ImageShape& from_shape, 
    const ImageShape& to_shape 
);


/** Convert pixel indices (y first, x second) to points (x first, y second),
    optionally centering on the pixel (+0.5). */
Points indices_to_points(const Indices2D& indices, bool center_pixel);


/** Simplify a path via the Ramer-Douglas-Peucker algorithm within epsilon pixels*/
Path rdp_line_simplification(const Path& path, double epsilon);

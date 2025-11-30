#pragma once

#include <array>
#include <optional>
#include <utility>
#include <vector>


typedef std::array<double, 2> Point;
typedef std::array<double, 2> Vector;
typedef std::vector<Point>    Points;
typedef std::vector<Point>    Path;
typedef std::vector<Path>     Paths;
typedef std::pair<Path, Path> PathPair;
typedef std::vector<PathPair> PairedPaths;
typedef std::pair<int,int>    ImageShape;

struct Box {
    double x0; // left
    double y0; // top
    double x1; // right
    double y1; // bottom
};


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



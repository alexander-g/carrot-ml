#pragma once

#include <array>
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



/** Test if points lie inside of a polygon */
std::vector<bool> points_in_polygon(const Points& points, const Path& polygon);


#ifndef POSTPROCESSING_HPP
#define POSTPROCESSING_HPP

#include <vector>

#include "../wasm-morpho/src/morphology.hpp"


typedef std::array<double, 2> Point;
typedef std::array<double, 2> Vector;
typedef std::vector<Point>    Points;
typedef std::vector<Point>    Path;
typedef std::vector<Path>     Paths;
typedef std::pair<int,int>    ImageShape;

Paths merge_paths(
    const Paths&      paths, 
    const ImageShape& imageshape,
    double max_distance = 0.30, 
    int    ray_width    = 50, 
    double max_overlap  = 0.3, 
    double min_length   = 0.05
) ;

#endif
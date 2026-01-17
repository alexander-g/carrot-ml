#include <cmath>
#include <deque>
#include <iostream>
#include <list>
#include <ranges>
#include <vector>
#include <utility>

#include "./postprocessing.hpp"

#include "./image-utils.hpp"
#include "./utils.hpp"

#include "../wasm-big-image/src/png-io.hpp"



typedef struct LineCoeffs {
    // x
    double a;
    // y
    double b;
    // offset
    double c;
} LineCoeffs;

typedef std::pair<int,int> IntPair;




const Point INFPOINT = {INFINITY, INFINITY};



/** Get points that are at most `distance` away from the reference point */
Points get_neighborhood(
    const Points& points, 
    const Point&  p, 
    double threshold
) {
    Path output;
    for(const Point& p_i: points)
        if( distance(p, p_i) < threshold )
            output.push_back(p_i);
    return output;
}



std::optional<Point> furthest_point(const Points& points, const Point& p) {
    if(points.size() == 0)
        return std::nullopt;

    double max_distance = 0;
    Point  max_p = p;
    for(const Point& p_i: points){
        const double distance_i = distance(p, p_i);
        if(distance_i > max_distance){
            max_distance = distance_i;
            max_p = p_i;
        }
    }
    return max_p;
}

/** Closest point to `p` at distance `d`. */
std::optional<Point> closest_point(
    const Points& points, 
    const Point&  p,
    double d = 0.0
) {
    if(points.size() == 0)
        return std::nullopt;

    double min_distance = INFINITY;
    std::optional<Point> min_p = std::nullopt;
    for(const Point& p_i: points){
        const double distance_i = std::abs(distance(p, p_i) - d);
        if(distance_i < min_distance){
            min_distance = distance_i;
            min_p = p_i;
        }
    }
    return min_p;
}


std::optional<std::vector<double>> paired_distances(
    const Points& points0, 
    const Points& points1
) {
    if(points0.size() != points1.size())
        return std::nullopt;

    std::vector<double> output;
    output.reserve(points0.size());
    for(int i = 0; i < points0.size(); i++)
        output.push_back( distance(points0[i], points1[i]) );
    return output;
}




/** Normalize x to unit length */
Vector normalize(const Vector& v) {
    double xlen = distance(v, Vector{0,0});
    xlen = std::max({xlen, 1e-6});
    return Vector{v[0] / xlen, v[1] / xlen};
}


/** Compute the coefficients of a line going through the points `p0` and `p1 */
LineCoeffs line_from_two_points(const Point& p0, const Point& p1) {
    const Vector direction = normalize( {p0[0] - p1[0], p0[1] - p1[1]} );
    const Vector ortho  = {direction[1], -direction[0]};
    const double offset = -(p0[0] * ortho[0] + p0[1] * ortho[1]);
    return LineCoeffs{ortho[0], ortho[1], offset};
}


/** Compute the coefficients of a line going through an endpoint `p` 
    of a set of points*/
std::optional<LineCoeffs> line_from_endpoint(
    const Point&  p, 
    const Points& points, 
    double threshold
) {
    const Points nhood = get_neighborhood(points, p, threshold);
    const auto a0 = average_points(nhood);
    if(!a0)
        return std::nullopt;
    
    const auto a1 = furthest_point(nhood, a0.value());
    if(!a1)
        return std::nullopt;
    
    return line_from_two_points(a0.value(), a1.value());
}



/** Evaluate the equation ax + by + c.*/
double eval_implicit_equation(const LineCoeffs& coef, const Point& p) {
    const Vector normcoef = normalize({coef.a, coef.b});
    return p[0] * normcoef[0]  +  p[1] * normcoef[1]  +  coef.c;
}

/** Rotate a line by 90° counter-clockwise so that it goes through point `p` */
LineCoeffs rotate_ccw(const LineCoeffs& coef, const Point& p) {
    const double a = -coef.b;
    const double b =  coef.a;
    const double c = -(p[0] * a  +  p[1] * b);
    return LineCoeffs{a, b, c};
}

LineCoeffs rotate_cw(const LineCoeffs& coef, const Point& p) {
    const double a =  coef.b;
    const double b = -coef.a;
    const double c = -(p[0] * a + p[1] * b);
    return LineCoeffs{a, b, c};
}



/** Approximate two points in a set that are the furthest away from each other */
std::optional<std::pair<Point, Point>>
get_endpoints_of_set_of_points(const Points& points) {
    if(points.size() == 0)
        return std::nullopt;
    
    const Point a0 = {0,0};
    const Point a1 = furthest_point(points, a0).value();
    const Point a2 = furthest_point(points, a1).value();
    return std::pair<Point,Point>{a2, a1};
}


Points project_points_onto_line(const LineCoeffs& coef, const Points& points){
    Points result;
    result.reserve(points.size());

    for(const Point& p: points){
        const double signed_distance = eval_implicit_equation(coef, p);
        const Vector direction = normalize({coef.a, coef.b});

        result.push_back({
            p[0] - direction[0] * signed_distance,
            p[1] - direction[1] * signed_distance
        });
    }
    return result;
}


/** Compute how much distribution `a` is overlapped by `b` */
std::optional<double> overlap_1d(
    const std::vector<double>& a, 
    const std::vector<double>& b
) {
    if(a.size() == 0 || b.size() == 0)
        return std::nullopt;
    
    const double a_min = *std::min_element(a.begin(), a.end());
    const double a_max = *std::max_element(a.begin(), a.end());
    const double b_min = *std::min_element(b.begin(), b.end());
    const double b_max = *std::max_element(b.begin(), b.end());

    if(a_min == a_max)
        return std::nullopt;

    const double overlap_start = std::max(a_min, b_min);
    const double overlap_end   = std::min(a_max, b_max);
    const double overlap_len   = std::max(0.0, overlap_end - overlap_start);

    return overlap_len / (a_max - a_min);
}


/** Approximate how much two paths overlap if projected onto each other */
std::optional<double> max_mutual_overlap(const Path& path0, const Path& path1) {
    if(path0.size() == 0 || path1.size() == 0)
        return std::nullopt;
    
    const LineCoeffs coef0 = line_from_two_points(path0.front(), path0.back());
    const LineCoeffs coef1 = line_from_two_points(path1.front(), path1.back());

    const Path path0_on_line_0 = project_points_onto_line(coef0, path0);
    const Path path1_on_line_0 = project_points_onto_line(coef0, path1);
    const Path path0_on_line_1 = project_points_onto_line(coef1, path0);
    const Path path1_on_line_1 = project_points_onto_line(coef1, path1);

    const Point p0 = get_endpoints_of_set_of_points(Points{
        path0_on_line_0.front(),
        path0_on_line_0.back(),
        path1_on_line_0.front(),
        path1_on_line_0.back(),
    }).value().first;  // NOTE: cant be std::nullopt because not empty
    const Point p1 = get_endpoints_of_set_of_points(Points{
        path0_on_line_1.front(),
        path0_on_line_1.back(),
        path1_on_line_1.front(),
        path1_on_line_1.back(),
    }).value().first;  // NOTE: cant be std::nullopt because not empty

    //const Point& p0 = path0_on_line_0.front();
    //const Point& p1 = path1_on_line_1.front();

    const auto distances0_on_0 = points_to_point_distances(path0_on_line_0, p0);
    const auto distances1_on_0 = points_to_point_distances(path1_on_line_0, p0);
    const auto distances0_on_1 = points_to_point_distances(path0_on_line_1, p1);
    const auto distances1_on_1 = points_to_point_distances(path1_on_line_1, p1);

    return std::max( 
        overlap_1d(distances0_on_0, distances1_on_0),
        overlap_1d(distances1_on_1, distances0_on_1)
    );

}




/** Compute which points are within `threshold` from line and `distance` 
    away in front of `p`. Returns indices of the points. */
std::vector<size_t> get_argpoints_in_ray (
    const LineCoeffs& coef, 
    const Point&  p, 
    const Points& points, 
    double threshold = 50, 
    double distance  = INFINITY
) {
    const LineCoeffs coef_inv = rotate_ccw(coef, p);
    
    std::vector<size_t> output;
    for(size_t i = 0; i < points.size(); i++) {
        const Point& p_i = points[i];
        // signed distance from `p` along the line
        const double signed_distance = eval_implicit_equation(coef_inv, p_i);
        if(signed_distance < 0 || signed_distance > distance)
            continue;
        
        const double distance_to_line = abs(eval_implicit_equation(coef, p_i));
        if(distance_to_line > threshold)
            continue;
        
        output.push_back(i);
    }
    return output;
}


/** Compute which points are within `threshold` from line and `distance` 
    away in front of `p` */
Points get_points_in_ray(
    const LineCoeffs& coef, 
    const Point&  p, 
    const Points& points, 
    double threshold = 50, 
    double distance  = INFINITY
) {
    const std::vector<size_t> indices = 
        get_argpoints_in_ray(coef, p, points, threshold, distance);
    
    Points output;
    output.reserve(indices.size());
    for(const size_t i: indices)
        output.push_back(points[i]);
    
    return output;
}


typedef std::vector<Path*> PathPointers;

PathPointers get_paths_in_ray(
    const LineCoeffs& coef, 
    const Point& p, 
          Paths& paths, 
    double threshold = 50, 
    double distance  = INFINITY
) {
    PathPointers result = {};
    for(Path& path: paths){
        const Points inraypoints = 
            get_points_in_ray(coef, p, path, threshold, distance);
        if( inraypoints.size() > 0 )
            result.push_back( &path );
    }
    return result;
}






Path merge_and_reorder(const Path& path0, const Path& path1) {
    const Point& p0_start = path0.front();
    const Point& p0_end   = path0.back();
    const Point& p1_start = path1.front();
    const Point& p1_end   = path1.back();

    const double d0 = distance(p0_start, p1_start);
    const double d1 = distance(p0_start, p1_end);
    const double d2 = distance(p0_end,   p1_start);
    const double d3 = distance(p0_end,   p1_end);

    const auto distances = {
        distance(p0_start, p1_start),
        distance(p0_start, p1_end),
        distance(p0_end,   p1_start),
        distance(p0_end,   p1_end)
    };
    const auto begin_it  = std::begin(distances);
    const auto argmin_it = std::min_element(begin_it, std::end(distances));
    
    const auto path1_rev = std::views::reverse(path1);
    return (argmin_it == begin_it    ) ? concat_copy(path1_rev, path0) 
         : (argmin_it == begin_it + 1) ? concat_copy(path1,     path0)
         : (argmin_it == begin_it + 2) ? concat_copy(path0,     path1)
         :                               concat_copy(path0,     path1_rev);

}



Paths merge_paths(
    const Paths&      paths, 
    const ImageShape& imageshape,
    double max_distance, 
    int    ray_width, 
    double max_overlap, 
    double min_length
) {
    Paths sorted_paths = sort_by_length(paths);

    if(max_distance < 1)
        //relative to image width (normally the smaller side of an image)
        max_distance *= std::min({imageshape.first, imageshape.second});
    if(min_length < 1)
        //relative to image width (normally the smaller side of an image)
        min_length *= std::min({imageshape.first, imageshape.second});


    for(int i = 0; i < sorted_paths.size(); i++) {
        Path& path = sorted_paths[i];
        const size_t len = path.size();

        if(len < min_length) { 
            // discard small paths, set to zero size to make sure it's skipped
            path.resize(0);
            continue;
        }

        std::deque<Point> endpoints{path.front(), path.back()};
        while( endpoints.size() > 0 ) {
            const Point e(endpoints.front());
            endpoints.pop_front();

            std::optional<Path*> closest_suitable_path = std::nullopt;
            double closest_suitable_path_distance = INFINITY;
            for(const double d: {len*0.1, len*0.2, len*1.0}){
                const auto coef = line_from_endpoint(e, path, d);
                if(!coef)
                    continue;
                
                // get paths that intersect the ray within a threshold
                PathPointers intersecting_paths = get_paths_in_ray(
                    coef.value(), 
                    e, 
                    sorted_paths, 
                    /*threshold=*/ray_width, 
                    /*distance= */max_distance
                );
                for(Path* isecpath_p: intersecting_paths){
                    // ignore candidates that have a large overlap
                    if( max_mutual_overlap(path, *isecpath_p) > max_overlap )
                        continue;
                    
                    const double path_to_endpoint_distance = 
                        distance(closest_point(*isecpath_p, e).value_or(INFPOINT), e);
                    
                    if(path_to_endpoint_distance < closest_suitable_path_distance) {
                        closest_suitable_path_distance = path_to_endpoint_distance;
                        closest_suitable_path = isecpath_p;
                    }
                }
            }

            if(!closest_suitable_path)
                continue;
            
            const Path merged_path =
                merge_and_reorder(path, *closest_suitable_path.value());
                
            sorted_paths[i] = merged_path;

            // #set size to zero to indicate that this path has been processed
            closest_suitable_path.value()->resize(0);
            // current path has changed: repeat iteration
            endpoints = {merged_path.front(), merged_path.back()};
        }
    }
    remove_zero_sized(sorted_paths);
    return sorted_paths;
}




/** Return unique int values in a std::vector<IntPair */
std::vector<int> unique_pairs_values(const std::vector<IntPair>& x) {
    std::unordered_map<int,int> count;
    count.reserve(x.size()*2);
    for(const auto& p : x) {
        count[p.first]++;
        count[p.second]++;
    }
    std::vector<int> result;
    result.reserve(count.size());
    for(const auto& kv: count) {
        if(kv.second == 1) 
            result.push_back(kv.first);
    }
    return result;
}



// TODO: not a good implementation, can result in duplicates
/** Select n roughly equidistant points on a path */
std::optional<Points> sample_points_on_path(const Path& path, int n) {
    if(path.size() == 0)
        return std::nullopt;
    
    const Point& a0 = path.front();
    const Point& a1 = path.back();
    const double D  = distance(a0, a1);

    Points result;
    result.reserve(n);
    for(const double d: linspace(0.0, D, n))
        // closest point at distance d
        // NOTE: cannot be std::nullopt bc path not empty
        result.push_back( closest_point(path, a0, d).value() );
    
    return result;
}

/** Robustly determine the next tree ring boundary relative to the boundary `l` */
std::optional<int> find_next_boundary(
    const Paths& paths, 
    const Path&  path, 
    bool         reverse
) {
    const Points sampled_points = 
        sample_points_on_path(path, 25+1).value_or(Points{});
    std::vector<int> sampled_path_indices;
    
    for(int i = 0; i < sampled_points.size() - 1; i++){
        const Point& p0 = sampled_points[reverse? i+1 : i];
        const Point& p1 = sampled_points[reverse? i : i+1];

        // fit a line
        const LineCoeffs coef = line_from_two_points(p0, p1);
        // rotate it by 90 deg
        const LineCoeffs coef_ortho = rotate_ccw(coef, p0);
        
        // find points from other paths that intersect the 90deg line
        // and note down the path with the closest point
        std::optional<int> closest_path_index = std::nullopt;
        double closest_path_distance = INFINITY;
        for(int j = 0; j < paths.size(); j++){
            const Path& other_path = paths[j];
            if(&other_path == &path)
                continue;
            
            const Points intersection_points = 
                get_points_in_ray(coef_ortho, p0, other_path, /*threshold=*/25);
            if(intersection_points.size() == 0) 
                continue;
            
            const double closest = *closest_distance(intersection_points, p0);
            if(closest < closest_path_distance){
                closest_path_distance = closest;
                closest_path_index = j;
            }
        }
        
        if(closest_path_index)
            sampled_path_indices.push_back(closest_path_index.value());
    }

    if(sampled_path_indices.size() == 0)
        return std::nullopt;
    
    // #take the most common path
    const int most_common_path_index = most_common(sampled_path_indices).value();
    return most_common_path_index;
}



std::vector<IntPair> find_longest_chain(std::vector<IntPair> pairs) {
    const std::vector<int> endpoints = unique_pairs_values(pairs);
    
    // for each endpoint construct a chain of integers
    std::vector<std::vector<IntPair>> chains;
    for(const int e: endpoints){
        int next_index = e;
        std::vector<IntPair> chain;

        // copy from vector into list, for easier manipulation
        std::list<IntPair> pairs_list(pairs.begin(), pairs.end());
        while(!pairs_list.empty()) {
            bool found = false;
            for(auto it = pairs_list.begin(); it != pairs_list.end(); it++) {
                IntPair pair = *it;
                
                if(pair.second == next_index)
                    pair = {pair.second, pair.first};
                
                if(pair.first == next_index) {
                    chain.push_back({pair.first, pair.second});
                    it = pairs_list.erase(it);
                    next_index = pair.second;
                    found = true;
                    break;
                }
            }
            
            if(!found)
                break; // just in case to avoid infinite loop
        }
        if(!chain.empty())
            chains.push_back(chain);
    }
    if(chains.empty())
        return {};

    // TODO: inefficient
    const auto longest_chain = sort_by_length(chains).front();
    return longest_chain;
}


/** Group tree ring boundaries into tuples */
std::vector<IntPair>  associate_boundaries(const Paths& paths) {
    if(paths.size() == 0)
        return {};
    
    std::vector<IntPair> pairs;
    for(int i = 0; i < paths.size(); i++){
        const Path& this_path = paths[i];
        const std::optional<int> next_of_this = 
            find_next_boundary(paths, this_path, /*reverse=*/false);
        if(!next_of_this)
            continue;
        
        const std::optional<int> prev_of_next = 
            find_next_boundary(paths, paths[*next_of_this], /*reverse=*/true);
        if(!prev_of_next)
            continue;

        // cycle-consistency
        if(i == prev_of_next.value())
            pairs.push_back({i, next_of_this.value()});
    }

    
    auto longest_chain = find_longest_chain(pairs);
    if(longest_chain.empty())
        return {};

    // reverse boundaries if needed, closest to the topleft corner first
    const Path& path_first = paths[longest_chain.front().first];
    const Path& path_last  = paths[longest_chain.back().second];
    const double meandist0 = 
        mean( points_to_point_distances(path_first, {0,0}) ).value_or(0.0);
    const double meandist1 = 
        mean( points_to_point_distances(path_last, {0,0}) ).value_or(0.0);

    if(meandist0 > meandist1) {
        std::reverse(longest_chain.begin(), longest_chain.end());
        for(auto &p : longest_chain) 
            std::swap(p.first, p.second);
    }

    return longest_chain;
}




double path_length(const Path& path) {
    double output = 0.0;
    if(path.size() < 2)
        return output;
    
    for(int i = 0; i < path.size() - 1; i++)
        output += distance(path[i], path[i+1]);
    
    return output;
}

Point interpolate_points(const Point& p0, const Point& p1, double alpha) {
    const Vector direction = {p1[0] - p0[0], p1[1] - p0[1]};
    return { 
        p0[0] + direction[0] * alpha, 
        p0[1] + direction[1] * alpha 
    };
}


Path resample_path(const Path& path, double step) {
    if(path.empty())
        return {};
    
    const double totallength = path_length(path);    
    Path output{path.front()};
    output.reserve(totallength / step);

    int i = 0;
    double position_i   = 0.0;
    double lastposition = 0.0;
    while(position_i < totallength && i < path.size()-1 ) {
        const double nextposition = lastposition + step;
        const double distance_to_next_point = distance(path[i], path[i+1]);
        const double position_i_plus_1 = position_i + distance_to_next_point;
        if(position_i_plus_1 < nextposition) {
            position_i += distance_to_next_point;
            i++;
            continue;
        }
        //else

        const double alpha = 
            (nextposition - position_i) / distance_to_next_point;
        const Point sampled_p = interpolate_points(path[i], path[i+1], alpha);
        output.push_back(sampled_p);
        
        lastposition = nextposition;
    }

    if( totallength - lastposition > 0.1  )
        output.push_back(path.back());
    return output;
}

Paths resample_paths(const Paths& paths) {
    Paths output;
    for(const auto& path: paths) {
        const double step = path_length(path) / 20;
        output.push_back( resample_path(path, step) );
    }
    return output;
}



/** Group points from path 0 to corresponding points from path 1 */
PathPair associate_pathpoints(const Path& path0, const Path& path1) {
    Path resampledpath0 = path0, resampledpath1 = path1;

    const bool flipped = (resampledpath0.size() < resampledpath1.size());
    if(flipped)
        std::swap(resampledpath0, resampledpath1);


    // find best offset
    double best_mean_distance = INFINITY;
    std::optional<int> best_offset = std::nullopt;
    for(int i = 0; i < resampledpath0.size() - resampledpath1.size() +1; i++) {
        const auto slicedpath0 = 
            slice_vector(resampledpath0, i, resampledpath1.size());
        const auto distances = paired_distances(slicedpath0, resampledpath1);
        if(distances){
            const double mean_distance = *mean( *distances );
            if(mean_distance < best_mean_distance){
                best_mean_distance = mean_distance;
                best_offset = i;
            }
        }
    }
    // should be always true
    if(best_offset)
        resampledpath0 = 
            slice_vector(resampledpath0, *best_offset, resampledpath1.size());

    if(flipped)
        std::swap(resampledpath0, resampledpath1);

    return {resampledpath0, resampledpath1};
}



std::vector<int> path_from_leaf(int leaf, const std::vector<int>& predecessors) {
    std::vector<int> path{ leaf };
    while(1){
        if( leaf < 0 || leaf >= predecessors.size())
            break;

        leaf = predecessors[leaf];
        path.push_back(leaf);
    }
    return path;
}

static std::vector<int> combine_paths(
    std::vector<int> path0,  //copy
    std::vector<int> path1  // copy
) {
    while(!path0.empty() && !path1.empty() && path0.back() == path1.back()) {
        path0.pop_back();
        path1.pop_back();
    }
    
    const auto path1_rev = std::views::reverse(path1);
    return concat_copy(path0, path1_rev) ;

}


std::optional<std::vector<int>> longest_path_from_dfs_result(const DFS_Result& dfs) {
    std::vector<std::vector<int>> paths;
    for(const int leaf: dfs.leaves) {
        // path from leaves to root of dfs
        const std::vector<int> path = path_from_leaf(leaf, dfs.predecessors);
        paths.push_back(path);
    }

    size_t longest_path_size = 0;
    std::optional<std::vector<int>> longest_path = std::nullopt;
    for(const auto& path0: paths){
        // consider single paths from leaf to root
        if(path0.size() > longest_path_size){
            longest_path = path0;
            longest_path_size = path0.size();
        }

        // as well as combined with another one, if root is not endpoint
        for(const auto& path1: paths){
            if(&path0 == &path1)
                continue;
            
            const auto combined = combine_paths(path0, path1);
            if(combined.size() > longest_path_size){
                longest_path = combined;
                longest_path_size = combined.size();
            }
        }
    }

    return longest_path;
}


std::vector<Indices2D> reorient_paths(const std::vector<Indices2D>& paths) {
    if(paths.size() == 0)
        return {};
    
    std::vector<Indices2D> valid_paths;
    std::vector<Vector> directions;
    std::vector<int> largest_axes;
    for(const Indices2D& path: paths) {
        if(path.size() < 2)
            continue;
        
        valid_paths.push_back(path);
        const Vector direction{ 
            (double)path.front().i - path.back().i, 
            (double)path.front().j - path.back().j
        };
        directions.push_back(direction);

        const int largest_axis = (direction[0] > direction[1])? 0 : 1;
        largest_axes.push_back(largest_axis);
    }
    if(largest_axes.empty())
        return {};
    
    const int common_axis = most_common(largest_axes).value();

    std::vector<int> orientations;
    for(const Vector& direction: directions)
        orientations.push_back( std::copysign(1, direction[common_axis]) );
    
    const int common_orientation = most_common(orientations).value();
    
    std::vector<Indices2D> new_paths;
    for(int i = 0; i < valid_paths.size(); i++) {
        Indices2D& path = valid_paths[i];
        if(orientations[i] == common_orientation)
            std::reverse(path.begin(), path.end());

        new_paths.push_back( path );
    }
    return new_paths;
}


std::vector<Index2D> gather_path_coordinates(
    const std::vector<int>&     path,
    const std::vector<Index2D>& coordinates
) {
    std::vector<Index2D> path_coordinates;
    for(const int i: path){
        if( i < 0 )
            continue;
        // TODO: how to handle i > coordinates.size()?
        path_coordinates.push_back(coordinates[i]);
    }
    return path_coordinates;
}


Paths indices_to_points(const std::vector<Indices2D>& indicesvector) {
    Paths output;
    for(const Indices2D& indices: indicesvector){
        Path path;
        for(const Index2D& index: indices)
            // yx to xy
            path.push_back({ (double)index.j, (double)index.i });
        
        output.push_back(path);
    }
    return output;
}


Paths segmentation_to_paths(
    const EigenBinaryMap& mask, 
    double min_length
) {
    const EigenBinaryMap skeleton = skeletonize(mask);
    const CCResult ccresult = connected_components(skeleton);

    
    if(min_length < 1.0){
        // relative to image width (normally the smaller side of an image)
        min_length *= std::min({mask.dimension(0), mask.dimension(1)});
    }

    std::vector<Indices2D> paths;
    for(const auto& dfs: ccresult.dfs_results){
        const auto path = longest_path_from_dfs_result(dfs);
        if(path && path->size() > 1 && path->size() > min_length)
            paths.push_back(  gather_path_coordinates(*path, dfs.visited)  );
    }

    const std::vector<Indices2D> reoriented_paths = reorient_paths(paths);
    return indices_to_points(reoriented_paths);
}


std::optional<BoxXYWH> bounding_box_of_indices(const Indices2D& indices) {
    if(indices.empty())
        return std::nullopt;

    int32_t y0 = 0x0fffffff, x0 = 0x0fffffff, y1 = 0, x1 = 0;
    for(const Index2D& index: indices) {
        if(y0 > index.i)
            y0 = index.i;
        
        if(x0 > index.j)
            x0 = index.j;

        if(y1 < index.i)
            y1 = index.i;

        if(x1 < index.j)
            x1 = index.j;
    }
    return BoxXYWH{
        .x = x0, 
        .y = y0, 
        .w = (uint32_t)x1 - x0 + 1, 
        .h = (uint32_t)y1 - y0 + 1
    };
}

Indices2D add_offset_to_indices(
    const Indices2D& indices, 
    const Index2D offset
) {
    Indices2D output;
    output.reserve(indices.size());
    for(const Index2D& index: indices)
        output.push_back(Index2D{
            .i = (Eigen::Index)((int)index.i + (int)offset.i), 
            .j = (Eigen::Index)((int)index.j + (int)offset.j), 
        });
    return output;
}


struct RasterizedComponent {
    EigenBinaryMap mask;
    Index2D offset;
};

std::optional<RasterizedComponent> rasterize_indices(const Indices2D& indices) {
    const auto expect_bbox = bounding_box_of_indices(indices);
    if(!expect_bbox)
        return std::nullopt;
    const BoxXYWH& bbox = expect_bbox.value();

    EigenBinaryMap mask(bbox.h, bbox.w);
    mask.setZero();
    for(const Index2D& index: indices)
        mask(index.i - bbox.y, index.j - bbox.x) = 1;

    return RasterizedComponent{mask, {bbox.y, bbox.x}};
}

Paths connected_components_to_paths(const ListOfIndices2D& list_of_cc_indices) {
    std::vector<Indices2D> paths;

    for(const Indices2D& cc_indices: list_of_cc_indices) {
        const auto expect_rasterized = rasterize_indices(cc_indices);
        if(!expect_rasterized)
            continue;
        const EigenBinaryMap& mask = expect_rasterized->mask;

        const EigenBinaryMap skeleton_mask = skeletonize(mask);
        const CCResult ccresult = connected_components(skeleton_mask);

        // should be a single iteration
        for(const auto& dfs: ccresult.dfs_results){
            const auto path = longest_path_from_dfs_result(dfs);
            if(path && path->size() > 1)
                paths.push_back( 
                    add_offset_to_indices(
                        gather_path_coordinates(*path, dfs.visited),
                        expect_rasterized->offset
                    )
                );
        }
    }
    
    const std::vector<Indices2D> reoriented_paths = reorient_paths(paths);
    return indices_to_points(reoriented_paths);
}









std::expected<TreeringsPostprocessingResult, std::string> 
postprocess_treeringmapfile(
    size_t      filesize,
    const void* read_file_callback_p,
    const void* read_file_handle,
    // shape: height first, width second
    const ImageShape& workshape,
    const ImageShape& og_shape,
    // flag to skip resizing mask, takes too long in the browser
    bool do_not_resize_to_og_shape
) {
    // if not png: error?

    // non-const for std::move
    auto expect_mask_and_cc = load_binary_png_connected_components_and_resize(
        filesize,
        read_file_callback_p,
        read_file_handle,
        {.width = (uint32_t)workshape.second, .height = (uint32_t)workshape.first}
    );
    if(!expect_mask_and_cc)
        return std::unexpected(expect_mask_and_cc.error());
    const EigenBinaryMap& mask = expect_mask_and_cc->mask;
    const ListOfIndices2D& objectpixels = expect_mask_and_cc->objects;
    
    const Paths simple_paths = connected_components_to_paths(objectpixels);
          Paths merged_paths = merge_paths(simple_paths, workshape);

    if(og_shape != workshape)
        merged_paths = scale_list_of_points(merged_paths, workshape, og_shape);

    const std::vector<IntPair> ring_labels = associate_boundaries(merged_paths);

    merged_paths = resample_paths(merged_paths);
    PairedPaths paired_paths;
    for(const IntPair& labelpair: ring_labels)
        paired_paths.push_back(
            associate_pathpoints(
                merged_paths[labelpair.first], 
                merged_paths[labelpair.second]
            )
        );
    

    // TODO: ring_areas = [treering_area(*rp) for rp in ring_points]

    const std::expected<Buffer_p, int> expect_treeringmap_workshape_png = 
        png_compress_image(
            to_uint8_p(mask.data()), 
            /*width=*/    mask.dimension(1),
            /*height=*/   mask.dimension(0),
            /*channels=*/ 1
        );
    if(!expect_treeringmap_workshape_png)
        return std::unexpected("PNG compression (workshape) failed");
    const Buffer_p treeringmap_workshape_png = *expect_treeringmap_workshape_png;

    std::optional<Buffer_p> treeringmap_og_shape_png = std::nullopt;
    if(workshape == og_shape)
        treeringmap_og_shape_png = treeringmap_workshape_png;
    else if(!do_not_resize_to_og_shape) {
        const std::expected<Buffer_p, int> expect_treeringmap_og_shape_png = 
            resize_image_and_encode_as_png(
                binary_to_rgba(mask),
                {.width=(uint32_t)og_shape.second, .height=(uint32_t)og_shape.first}
            );
        if(!expect_treeringmap_og_shape_png)
            return std::unexpected("PNG compression (og shape) failed");
        treeringmap_og_shape_png = expect_treeringmap_og_shape_png.value();
    } //else dont resize here, takes too long

    return TreeringsPostprocessingResult{
        /*treeringmap_workshape_png = */ treeringmap_workshape_png,
        /*treeringmap_og_shape_png  = */ treeringmap_og_shape_png,
        /*ring_points_xy            = */ paired_paths
    };
}



#ifndef PYBIND_UTILS_HPP
#define PYBIND_UTILS_HPP

#include <pybind11/numpy.h>

#include "./postprocessing.hpp"


namespace py = pybind11;
typedef py::array_t<double, py::array::c_style | py::array::forcecast> py_f64_array;
typedef py::array_t<bool, py::array::c_style | py::array::forcecast> py_bool_array;


/** ndarray of shape [N,2] to std::vector<std::pair<double,double>> */
Path  path_numpy_to_stdvec(const py_f64_array& a);

/** list of ndarrays of shape [N,2] to std::vector<std::vector<...> */
Paths paths_numpy_to_stdvec(py::list paths);

/** std::vector<std::pair<double,double>> to ndarray shape [N,2] */
py_f64_array path_stdvec_to_numpy(const Path& path);

/** std::vector<std::vector<...> to list of ndarrays of shape [N,2] */
py::list paths_stdvec_to_numpy(const Paths& paths);

/** std::vector<std::pair<int,int>> to list of tp.Tuple[int,int] */
py::list vec_pairs_to_pylist(const std::vector<std::pair<int,int>>& v);


#endif
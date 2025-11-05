#ifndef PYBIND_UTILS_HPP
#define PYBIND_UTILS_HPP

#include <pybind11/numpy.h>

#include "./postprocessing.hpp"


namespace py = pybind11;


Paths paths_numpy_to_stdvec(py::list paths);
py::list paths_stdvec_to_numpy(const Paths& paths);


#endif
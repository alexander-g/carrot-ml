#include <iostream>
#include <utility>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "./src/pybind-utils.hpp"
#include "./src/postprocessing.hpp"


namespace py = pybind11;




py::list merge_paths_py(
    const py::list    paths_py, 
    const ImageShape& imageshape
) {
    const Paths paths = paths_numpy_to_stdvec(paths_py);
    const Paths merged_paths = merge_paths(paths, imageshape);

    return paths_stdvec_to_numpy(merged_paths);
}

py::list associate_boundaries_py(const py::list paths_py) {
    const Paths paths = paths_numpy_to_stdvec(paths_py);
    const auto chain  = associate_boundaries(paths);

    return vec_pairs_to_pylist(chain);
}

std::pair<py_f64_array, py_f64_array> associate_pathpoints_py(
    const py_f64_array path0_py, 
    const py_f64_array path1_py
) {
    const Path path0 = path_numpy_to_stdvec(path0_py);
    const Path path1 = path_numpy_to_stdvec(path1_py);

    const auto new_paths = associate_pathpoints(path0, path1);
    return {
        path_stdvec_to_numpy(new_paths.first), 
        path_stdvec_to_numpy(new_paths.second)
    };
}



PYBIND11_MODULE(carrot_postprocessing_ext, m) {
    m.doc() = "carrot postprocessing c++ extension";

    m.def(
        "merge_paths", 
        merge_paths_py, 
        py::arg("paths").noconvert(),
        py::arg("imageshape")
    );

    m.def(
        "associate_boundaries", 
        associate_boundaries_py, 
        py::arg("paths").noconvert()
    );

    m.def(
        "associate_pathpoints",
        associate_pathpoints_py,
        py::arg("path0").noconvert(),
        py::arg("path1").noconvert()
    );
}



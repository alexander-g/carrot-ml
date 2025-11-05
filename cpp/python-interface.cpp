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




PYBIND11_MODULE(carrot_postprocessing_ext, m) {
    m.doc() = "carrot postprocessing c++ extension";

    m.def(
        "merge_paths", 
        merge_paths_py, 
        py::arg("paths").noconvert(),
        py::arg("imageshape")
    );
}



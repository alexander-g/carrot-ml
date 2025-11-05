#include <iostream>

#include "./pybind-utils.hpp"


Paths paths_numpy_to_stdvec(py::list paths){
    Paths out;
    out.reserve(py::len(paths));
    
    for (py::handle h : paths) {
        py::array_t<double, py::array::c_style | py::array::forcecast> a =
            py::cast<py::array>(h);
        if (a.ndim() != 2 || a.shape(1) != 2)
            throw std::runtime_error("Path array not of shape [N,2]");
    
        const ssize_t n = a.shape(0);
        const auto buf  = a.unchecked<2>();
        Path path;
        path.reserve(n);
        for(int i = 0; i < n; i++)
            path.push_back({buf(i,0), buf(i,1)});
        
        out.push_back(std::move(path));
    }
    return std::move(out);
}


py::list paths_stdvec_to_numpy(const Paths& paths){
    py::list out;
    
    for (const Path& path : paths) {
        const ssize_t n = static_cast<ssize_t>(path.size());
        std::vector<ssize_t> shape = { n, 2 };
        py::array_t<double, py::array::c_style> a(shape);
        auto buf = a.mutable_unchecked<2>();
        for (ssize_t i = 0; i < n; ++i){
            buf(i, 0) = path[i][0];
            buf(i, 1) = path[i][1];
        }
        out.append(std::move(a));
    }
    return out;
}


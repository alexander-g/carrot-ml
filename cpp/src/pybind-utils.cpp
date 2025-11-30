#include <iostream>

#include "./pybind-utils.hpp"


Path path_numpy_to_stdvec(const py_f64_array& a) {
    if (a.ndim() != 2 || a.shape(1) != 2)
        throw std::runtime_error("Path array not of shape [N,2]");

    const ssize_t n = a.shape(0);
    const auto buf  = a.unchecked<2>();
    Path path;
    path.reserve(n);
    for(int i = 0; i < n; i++)
        path.push_back({buf(i,0), buf(i,1)});
    return path;
}


Paths paths_numpy_to_stdvec(py::list paths){
    Paths out;
    out.reserve(py::len(paths));
    
    for (py::handle h : paths) {
        const py_f64_array a = py::cast<py::array>(h);
        
        const Path path = path_numpy_to_stdvec(a);
        out.push_back(path);
    }
    return std::move(out);
}


py_f64_array path_stdvec_to_numpy(const Path& path) {
    const ssize_t n = static_cast<ssize_t>(path.size());
    const std::vector<ssize_t> shape = { n, 2 };
    py::array_t<double, py::array::c_style> a(shape);
    auto buf = a.mutable_unchecked<2>();
    for (ssize_t i = 0; i < n; ++i){
        buf(i, 0) = path[i][0];
        buf(i, 1) = path[i][1];
    }
    return a;
}

py_bool_array bool_stdvec_to_numpy(const std::vector<bool>& boolarray) {
    const int n = boolarray.size();
    py_bool_array output({n});
    auto buffer = output.mutable_unchecked<1>();
    for(int i = 0; i < n; i++)
        buffer(i) = boolarray[i];
    return output;
}

py::list paths_stdvec_to_numpy(const Paths& paths){
    py::list out;
    
    for (const Path& path : paths)
        out.append( path_stdvec_to_numpy(path) );
    return out;
}

py::list vec_pairs_to_pylist(const std::vector<std::pair<int,int>>& v) {
    py::list out;
    for (const auto &p : v) 
        out.append(py::make_tuple(p.first, p.second));

    return out;
}

py::list vec_paired_paths_to_numpy(const PairedPaths& pp) {
    py::list out;
    for(const auto& pathpair: pp )
        out.append( 
            py::make_tuple( 
                path_stdvec_to_numpy(pathpair.first),
                path_stdvec_to_numpy(pathpair.second)
            ) 
        );

    return out;
}


py::bytes buffer_to_bytes(const Buffer& b) {
    return py::bytes((const char*)b.data, (size_t)b.size);
}


py::list cell_info_to_py(const std::vector<CellInfo>& cell_info) {
    py::list out;
    for(const CellInfo& cell: cell_info){
        py::dict cell_py;
        cell_py["id"] = cell.id;
        cell_py["box_xy"] = py::make_tuple(
            cell.box_xy.x0, 
            cell.box_xy.y0, 
            cell.box_xy.x1, 
            cell.box_xy.y1
        );
        cell_py["year_index"]      = cell.year_index;
        cell_py["area_px"]         = cell.area_px;
        cell_py["position_within"] = cell.position_within;
        out.append(cell_py);
    }
        
    return out;
}



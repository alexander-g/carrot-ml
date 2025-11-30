#include <iostream>
#include <utility>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "./wasm-morpho/src/pybind-utils.hpp"
#include "./wasm-morpho/src/morphology.hpp"
#include "./wasm-big-image/src/util.hpp"
#include "./src/pybind-utils.hpp"
#include "./src/postprocessing.hpp"
#include "./src/postprocessing_cells.hpp"


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

py::list segmentation_to_paths_py(const py_bool_array& mask_py, double min_length) {
    const EigenBinaryMap mask = boolarray_to_eigen_tensor(mask_py);
    const Paths paths = segmentation_to_paths(mask, min_length);
    return paths_stdvec_to_numpy(paths);
}

py_bool_array points_in_polygon_py(
    const py_f64_array points_py, 
    const py_f64_array polygon_py
) {
    const Points points  = path_numpy_to_stdvec(points_py);
    const Path   polygon = path_numpy_to_stdvec(polygon_py);

    std::vector<bool> insides = points_in_polygon(points, polygon);
    return bool_stdvec_to_numpy(insides);
}



py::dict postprocess_treeringmapfile_py(
    const std::string& path, 
    const ImageShape& workshape, 
    const ImageShape& og_shape
) {
    const auto fhandle_o = FileHandle::open(path.c_str());
    if(!fhandle_o)
        throw std::runtime_error("Could not open file");
    const FileHandle* fhandle = fhandle_o.value().get();

    const auto output_x = postprocess_treeringmapfile(
        fhandle->size, 
        (const void*) &fhandle->read_callback, 
        (void*) fhandle, 
        workshape,
        og_shape
    );
    if(!output_x)
        throw std::runtime_error("Postprocessing failed");

    py::dict d;
    d["treeringmap_workshape_png"] = 
        buffer_to_bytes(*output_x->treeringmap_workshape_png);
    //d["treeringmap_ogshape_png"]   = "???";
    d["ring_points_xy"] = vec_paired_paths_to_numpy(output_x->ring_points_xy);
    return d;
}

py::dict postprocess_cellmapfile_py(
    const std::string& path, 
    const ImageShape& workshape, 
    const ImageShape& og_shape
) {
    const auto fhandle_o = FileHandle::open(path.c_str());
    if(!fhandle_o)
        throw std::runtime_error("Could not open file");
    const FileHandle* fhandle = fhandle_o.value().get();

    const auto output_x = postprocess_cellmapfile(
        fhandle->size, 
        (const void*) &fhandle->read_callback, 
        (void*) fhandle, 
        workshape,
        og_shape
    );
    if(!output_x)
        throw std::runtime_error("Postprocessing failed");

    py::dict d;
    d["cellmap_workshape_png"] = 
        buffer_to_bytes(*output_x->cellmap_workshape_png);
    d["instancemap_workshape_png"] = 
        buffer_to_bytes(*output_x->instancemap_workshape_png);
    return d;
}

py::dict postprocess_combined_from_files_py(
    const std::string& cellmappath,
    const std::string& treeringmappath,
    const ImageShape& workshape, 
    const ImageShape& og_shape
) {
    const auto expect_fhandle_cells = FileHandle::open(cellmappath.c_str());
    const auto expect_fhandle_rings = FileHandle::open(treeringmappath.c_str());
    if(!expect_fhandle_cells)
        throw std::runtime_error("Could not open cellmap file");
    if(!expect_fhandle_rings)
        throw std::runtime_error("Could not open treerings file");
    const FileHandle* fhandle_cells = expect_fhandle_cells.value().get();
    const FileHandle* fhandle_rings = expect_fhandle_rings.value().get();


    const auto expect_output_cells = postprocess_cellmapfile(
        fhandle_cells->size, 
        (const void*) &fhandle_cells->read_callback, 
        (void*) fhandle_cells, 
        workshape,
        og_shape
    );
    if(!expect_output_cells)
        throw std::runtime_error("Cells postprocessing failed");
    const CellsPostprocessingResult& output_cells = *expect_output_cells;

    const auto expect_output_rings = postprocess_treeringmapfile(
        fhandle_rings->size, 
        (const void*) &fhandle_rings->read_callback, 
        (void*) fhandle_rings, 
        workshape,
        og_shape
    );
    if(!expect_output_rings)
        throw std::runtime_error("Treerings postprocessing failed");
    const TreeringsPostprocessingResult& output_rings = *expect_output_rings;

    const auto expect_output_combined = 
        postprocess_combined(
            output_rings.ring_points_xy, 
            output_cells.cells, 
            workshape,
            og_shape
        );
    if(!expect_output_combined)
        throw std::runtime_error("Combined postprocessing failed");
    const CombinedPostprocessingResult& output_combined = *expect_output_combined;


    py::dict d;
    d["cellmap_workshape_png"] = 
        buffer_to_bytes(*output_cells.cellmap_workshape_png);
    d["instancemap_workshape_png"] = 
        buffer_to_bytes(*output_cells.instancemap_workshape_png);
    d["treeringmap_workshape_png"] = 
        buffer_to_bytes(*output_rings.treeringmap_workshape_png);
    d["ringmap_workshape_png"] =
        buffer_to_bytes(*output_combined.ringmap_rgb_png);
    d["cell_info"] = cell_info_to_py(output_combined.cell_info);
    return d;
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

    m.def(
        "segmentation_to_paths",
        segmentation_to_paths_py,
        py::arg("mask").noconvert(),
        py::arg("min_length")
    );

    m.def(
        "points_in_polygon",
        points_in_polygon_py,
        py::arg("p"),
        py::arg("polygon")
    );

    m.def(
        "postprocess_treeringmapfile",
        postprocess_treeringmapfile_py,
        py::arg("path").noconvert(),
        py::arg("workshape"),
        py::arg("og_shape")
    );

    m.def(
        "postprocess_cellmapfile",
        postprocess_cellmapfile_py,
        py::arg("path").noconvert(),
        py::arg("workshape"),
        py::arg("og_shape")
    );

    m.def(
        "postprocess_combined",
        postprocess_combined_from_files_py,
        py::arg("cellmappath").noconvert(),
        py::arg("treeringmappath").noconvert(),
        py::arg("workshape"),
        py::arg("og_shape")
    );
    
}



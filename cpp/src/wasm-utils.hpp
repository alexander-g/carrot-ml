#pragma once

#include <string>
#include <vector>

#include "./postprocessing.hpp"
#include "./postprocessing_cells.hpp"



std::string paired_paths_to_json(const PairedPaths& pp);

std::string cell_info_to_json(const std::vector<CellInfo>& cell_info);



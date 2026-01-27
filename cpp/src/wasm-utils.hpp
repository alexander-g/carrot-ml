#pragma once

#include <expected>
#include <string>
#include <vector>

#include "./postprocessing.hpp"
#include "./postprocessing_cells.hpp"



std::string paired_paths_to_json(const PairedPaths& pp);

std::string cell_info_to_json(const std::vector<CellInfo>& cell_info);

Buffer_p serialize_list_of_rle_components(const ListOfRLEComponents& list);
Buffer_p serialize_list_of_indices2d_as_rle(const ListOfIndices2D& list);

std::expected<std::vector<RLERun> , std::string> 
deserialize_list_of_rle(const uint8_t* data, uint32_t nbytes);

std::expected<AreaOfInterestRect, std::string>  parse_aoi_json(const std::string&);
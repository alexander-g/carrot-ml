#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include "./wasm-utils.hpp"


json path_to_json(const Path& path) {
    json output = json::array({});
    for(const Point& p: path) {
        output.push_back( {p[0], p[1]} );
    }
    return output;
}


std::string paired_paths_to_json(const PairedPaths& pp) {
    json output = json::array({});
    for(const PathPair& ppair: pp) {
        output.push_back({
            path_to_json(ppair.first),
            path_to_json(ppair.second)
        });
    }
    return output.dump();
}



json box_to_json(const Box& box) {
    return { box.x0, box.y0, box.x1, box.y1 };
}

std::string cell_info_to_json(const std::vector<CellInfo>& cell_info) {
    json output = json::array({});
    for(const CellInfo& cell: cell_info) {
        output.push_back({
            {"id",               cell.id},
            {"box_xy",           box_to_json(cell.box_xy)},
            {"year_index",       cell.year_index},
            {"area",             cell.area_px},
            {"position_within",  cell.position_within},
        });
    }
    return output.dump();
}



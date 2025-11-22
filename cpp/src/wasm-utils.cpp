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




#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include "./image-utils.hpp"
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


Buffer_p serialize_list_of_rle_components(const ListOfRLEComponents& list) {
    // non-const for std::move
    std::vector<RLERun> flat_rle = flatten_rle_components(list);
    const uint64_t nbytes = flat_rle.size() * sizeof(RLERun);
    uint8_t* data_p = (uint8_t*) flat_rle.data();
    
    // custom deleter that owns the vector flat_rle
    auto deleter = [rle = std::move(flat_rle)](Buffer* b) mutable {
        delete b;
    };
    return Buffer_p( new Buffer{data_p, nbytes}, std::move(deleter) );
}


Buffer_p serialize_list_of_indices2d_as_rle(const ListOfIndices2D& list) {
    const ListOfRLEComponents nested_rle = dense_to_rle_components(list);
    return serialize_list_of_rle_components(nested_rle);
}


std::expected<std::vector<RLERun> , std::string> 
deserialize_list_of_rle(const uint8_t* data, uint32_t nbytes) {
    if(nbytes == 0)
        return std::vector<RLERun>{};
    if(nbytes % sizeof(RLERun) != 0)
        return std::unexpected("number of bytes not divisible by size of RLERun");
    
    const uint32_t n = (uint32_t)(nbytes / (double)sizeof(RLERun));
    const RLERun* p  = (RLERun*) data;
    std::vector<RLERun>  output(p, p+n);
    return output;
}


std::expected<AreaOfInterestRect, std::string> 
parse_aoi_json(const std::string& buffer) {
    json j;
    try {
        j = json::parse(buffer);
    } catch (const json::parse_error &e) {
        return std::unexpected("JSON parse error: " + std::string(e.what()) );
    }

    if(!j.is_array() || j.size() != 4) 
        return std::unexpected("expected array of 4 elements");

    AreaOfInterestRect output;
    // future-proofing
    static_assert( sizeof(AreaOfInterestRect) == sizeof(double) * 8 );
    double* output_as_f64 = (double*) &output;
    for(int i = 0; i < j.size(); i++) {
        const json &item = j[i];
        if(!item.is_array() || item.size() != 2)
            return std::unexpected("Element "+std::to_string(i)+" not a 2d point");

        const json& v0 = item[0];
        const json& v1 = item[1];
        if(!v0.is_number() || !v1.is_number())
            return std::unexpected("Element "+std::to_string(i)+" not 2 numbers");

        output_as_f64[i*2 +0] = v0;
        output_as_f64[i*2 +1] = v1;
    }
    return std::move(output);
}

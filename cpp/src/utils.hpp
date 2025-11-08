#pragma once

#include <optional>



/** Sort vector of vectors in descending order  */
template<typename T>
std::vector<std::vector<T>> sort_by_length(std::vector<std::vector<T>> x) {
    std::ranges::sort(x, std::greater{}, &std::vector<T>::size);
    return x;
}




/** In-place remove empty elements in vector of vectors */
template<class T>
void remove_zero_sized(std::vector<std::vector<T>>& v) {
    v.erase(
        std::remove_if(
            v.begin(), 
            v.end(),
            [](const std::vector<T>& inner){ return inner.empty(); }),
            v.end()
        );
}


/** Slice vector, copy. */
template<typename T>
std::vector<T> slice_vector(const std::vector<T>& v, size_t start, size_t len){
    if(start > v.size()) 
        start = v.size();
    const size_t end = std::min(v.size(), start + len);
    return std::vector<T>(v.begin() + start, v.begin() + end);
}


/** Compute the average of vector elements */
template<typename T>
std::optional<double> mean(std::vector<T> x) {
    return std::accumulate(x.begin(), x.end(), 0.0) / x.size();
}


/** Generate n equidistant numbers from start to stop.  */
template<std::floating_point T>
std::vector<T> linspace(T start, T stop, std::size_t n, bool endpoint = true) {
    if(n == 0)
        return {};
    if(n == 1)
        return {start};
    std::vector<T> output;
    output.reserve(n);
    T step = endpoint ? (stop - start) / static_cast<T>(n - 1)
                      : (stop - start) / static_cast<T>(n);
    for(std::size_t i = 0; i < n; i++)
        output.push_back(start + step * static_cast<T>(i));
    if(endpoint)
        output.back() = stop; // avoid FP drift
    return output;
}

/** Find the most common element in a vector */
template<class T>
std::optional<T> most_common(const std::vector<T>& v) {
    if (v.empty()) 
        return std::nullopt;
    
    std::unordered_map<T, size_t> freq;
    freq.reserve(v.size());

    for(const auto& x : v) 
        ++freq[x];
    
    auto best = std::begin(freq);
    for (auto it = std::next(std::begin(freq)); it != std::end(freq); it++)
        if(it->second > best->second)
            best = it;
    return best->first;
}

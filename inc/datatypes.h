#ifndef DATATYPES_H
#define DATATYPES_H

#include <vector>
#include <unordered_map>
#include <string>

struct pair_hash
{
    template <class T1, class T2>
    std::size_t operator() (const std::pair<T1,T2> &p) const
    {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        // Mainly for demonstration purposes, i.e. works but is overly simple
        // In the real world, use something like boost.hash_combine
        return h1 ^ h2;
    }
};

// For mapping (a,m) pairs to some float
using Map = std::unordered_map<std::pair<std::string, std::string>, float, pair_hash>;

template<typename T>
using Matrix = std::vector<std::vector<T>>;

#endif
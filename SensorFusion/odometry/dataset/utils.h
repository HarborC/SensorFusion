#pragma once

#include <iostream>
#include <string>
#include <vector>

namespace DatasetIO {

inline void StringSplit(const std::string& str, const std::string& splits,
                        std::vector<std::string>& res) {
    if (str == "")
        return;
    std::string strs = str + splits;
    size_t pos = strs.find(splits);
    int step = splits.size();

    while (pos != strs.npos) {
        std::string temp = strs.substr(0, pos);
        res.push_back(temp);
        strs = strs.substr(pos + step, strs.size());
        pos = strs.find(splits);
    }
}

}  // namespace DatasetIO
#pragma once
#include <vector>
#include <string>
#include <unordered_map>

class Tokenizer {
private:
    std::vector<std::string> vocab;                     
    std::unordered_map<std::string, int> tokenToId;        
    std::vector<std::string> idToToken;                   
    int unknownId = -1;              

public:
    Tokenizer(const std::vector<std::string>& vocabList);

    std::vector<int> encode(const std::vector<std::string>& tokens);

    std::vector<std::string> decode(const std::vector<int>& ids);

    std::vector<std::vector<int>> batchEncode(const std::vector<std::vector<std::string>>& sentences, int numThreads = 2);

    std::vector<std::vector<std::string>> batchDecode(const std::vector<std::vector<int>>& encodedSentences, int numThreads = 2);
};

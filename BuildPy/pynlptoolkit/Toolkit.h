#pragma once
#include <string>
#include <vector>
#include <unordered_map>

class Toolkit {
public:
    // Tokenize
    static std::vector<std::string> tokenize(const std::string& text);

    // Bag-of-Words
    static std::unordered_map<std::string, int> getBagOfWords(const std::vector<std::string>& tokens, int numThreads = 2);

    // N-grams
    static std::vector<std::string> getNGrams(const std::vector<std::string>& tokens, int n);

    // Normalization
    static std::string toLower(const std::string& text);
    static std::string removePunctuation(const std::string& text);

    // Embeddings
    static std::unordered_map<std::string, std::vector<float>> getEmbeddings(const std::vector<std::string>& tokens, size_t vectorEmbeddingSize = 300, int numThreads = 2);

    // Stemming
    static std::string stem(const std::string& text);
};

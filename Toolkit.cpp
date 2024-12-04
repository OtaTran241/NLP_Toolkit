#include "Toolkit.h"
#include <sstream>
#include <algorithm>
#include <iostream>
#include <random>
#include <thread>
#include <future>
#include <mutex>

std::vector<std::vector<std::string>> splitTokens(const std::vector<std::string>& tokens, size_t numThreads) {
    size_t blockSize = tokens.size() / numThreads;
    std::vector<std::vector<std::string>> splitBlocks;
    for (size_t i = 0; i < numThreads; ++i) {
        size_t start = i * blockSize;
        size_t end = (i == numThreads - 1) ? tokens.size() : (i + 1) * blockSize;
        splitBlocks.emplace_back(tokens.begin() + start, tokens.begin() + end);
    }
    return splitBlocks;
}

std::vector<std::string> Toolkit::tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::istringstream stream(text);
    std::string word;

    while (stream >> word) {
        tokens.push_back(word);
    }
    return tokens;
}

std::unordered_map<std::string, int> Toolkit::getBagOfWords(const std::vector<std::string>& tokens, int numThreads) {
    size_t maxThreads = std::thread::hardware_concurrency();

    if (numThreads <= 0 || numThreads > static_cast<int>(maxThreads)) {
        numThreads = maxThreads;
    }

    std::vector<std::future<std::unordered_map<std::string, int>>> futures;
    auto splitBlocks = splitTokens(tokens, numThreads);

    std::mutex mtx;
    std::unordered_map<std::string, int> combinedResult;

    for (const auto& block : splitBlocks) {
        futures.push_back(std::async(std::launch::async, [&block]() {
            std::unordered_map<std::string, int> localBag;
            for (const auto& token : block) {
                localBag[token]++;
            }
            return localBag;
            }));
    }

    for (auto& future : futures) {
        auto localBag = future.get();
        std::lock_guard<std::mutex> lock(mtx);
        for (const auto& [token, count] : localBag) {
            combinedResult[token] += count;
        }
    }

    return combinedResult;
}

std::vector<std::string> Toolkit::getNGrams(const std::vector<std::string>& tokens, int n) {
    std::vector<std::string> ngrams;

    if (tokens.empty() || n <= 0) return ngrams;

    for (size_t i = 0; i + n <= tokens.size(); ++i) {
        std::ostringstream ngram;
        for (size_t j = i; j < i + n; ++j) {
            ngram << tokens[j];
            if (j < i + n - 1) ngram << " ";
        }
        ngrams.push_back(ngram.str());
    }
    return ngrams;
}

std::string Toolkit::toLower(const std::string& text) {
    std::string result = text;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

std::string Toolkit::removePunctuation(const std::string& text) {
    std::string result;
    for (char ch : text) {
        if (!std::ispunct(ch)) {
            result += ch;
        }
    }
    return result;
}

std::unordered_map<std::string, std::vector<float>> Toolkit::getEmbeddings(const std::vector<std::string>& tokens, size_t embeddingSize, int numThreads) {
    size_t maxThreads = std::thread::hardware_concurrency();

    if (numThreads <= 0 || numThreads > static_cast<int>(maxThreads)) {
        numThreads = maxThreads; 
    }

    std::vector<std::future<std::unordered_map<std::string, std::vector<float>>>> futures;
    auto splitBlocks = splitTokens(tokens, numThreads);
    std::unordered_map<std::string, std::vector<float>> combinedEmbeddings;
    std::mutex mtx;

    for (const auto& block : splitBlocks) {
        futures.push_back(std::async(std::launch::async, [&block, embeddingSize]() {
            std::unordered_map<std::string, std::vector<float>> localEmbeddings;
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(-1.0, 1.0);

            for (const auto& token : block) {
                std::vector<float> embedding(embeddingSize);
                for (auto& value : embedding) {
                    value = static_cast<float>(dis(gen));
                }
                localEmbeddings[token] = embedding;
            }
            return localEmbeddings;
            }));
    }

    for (auto& future : futures) {
        auto localEmbeddings = future.get();
        std::lock_guard<std::mutex> lock(mtx);
        combinedEmbeddings.insert(localEmbeddings.begin(), localEmbeddings.end());
    }

    return combinedEmbeddings;
}

std::string Toolkit::stem(const std::string& text) {
    if (text.size() > 4) {
        return text.substr(0, text.size() - 2);
    }
    return text;
}
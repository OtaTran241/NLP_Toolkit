#include "Tokenizer.h"
#include <stdexcept>
#include <thread>
#include <future>
#include <algorithm>

// Constructor
Tokenizer::Tokenizer(const std::vector<std::string>& vocabList) : vocab(vocabList), idToToken(vocabList) {
    for (size_t i = 0; i < vocab.size(); ++i) {
        tokenToId[vocab[i]] = static_cast<int>(i);
    }
    if (tokenToId.find("<UNK>") == tokenToId.end()) {
        vocab.push_back("<UNK>");
        idToToken.push_back("<UNK>");
        unknownId = static_cast<int>(vocab.size() - 1);
        tokenToId["<UNK>"] = unknownId;
    }
    else {
        unknownId = tokenToId["<UNK>"];
    }
}

// Encode a single sentence
std::vector<int> Tokenizer::encode(const std::vector<std::string>& tokens) {
    std::vector<int> encodedTokens;
    for (const auto& token : tokens) {
        if (tokenToId.find(token) != tokenToId.end()) {
            encodedTokens.push_back(tokenToId[token]);
        }
        else {
            encodedTokens.push_back(unknownId);
        }
    }
    return encodedTokens;
}

// Decode a single encoded sequence
std::vector<std::string> Tokenizer::decode(const std::vector<int>& ids) {
    std::vector<std::string> decodedTokens;
    for (const auto& id : ids) {
        if (id >= 0 && id < static_cast<int>(idToToken.size())) {
            decodedTokens.push_back(idToToken[id]);
        }
        else {
            throw std::out_of_range("Invalid token ID in decode.");
        }
    }
    return decodedTokens;
}

// Batch encode
std::vector<std::vector<int>> Tokenizer::batchEncode(const std::vector<std::vector<std::string>>& sentences, int numThreads) {
    size_t numSentences = sentences.size();
    numThreads = (numThreads == -1) ? std::thread::hardware_concurrency() : std::min(numThreads, static_cast<int>(numSentences));

    std::vector<std::future<std::vector<int>>> futures;
    std::vector<std::vector<int>> results(numSentences);

    for (size_t i = 0; i < numSentences; ++i) {
        futures.push_back(std::async(std::launch::async, [this, &sentences, i]() {
            return this->encode(sentences[i]);
            }));
    }

    for (size_t i = 0; i < numSentences; ++i) {
        results[i] = futures[i].get();
    }
    return results;
}

// Batch decode
std::vector<std::vector<std::string>> Tokenizer::batchDecode(const std::vector<std::vector<int>>& encodedSentences, int numThreads) {
    size_t numSentences = encodedSentences.size();
    numThreads = (numThreads == -1) ? std::thread::hardware_concurrency() : std::min(numThreads, static_cast<int>(numSentences));

    std::vector<std::future<std::vector<std::string>>> futures;
    std::vector<std::vector<std::string>> results(numSentences);

    for (size_t i = 0; i < numSentences; ++i) {
        futures.push_back(std::async(std::launch::async, [this, &encodedSentences, i]() {
            return this->decode(encodedSentences[i]);
            }));
    }

    for (size_t i = 0; i < numSentences; ++i) {
        results[i] = futures[i].get();
    }
    return results;
}

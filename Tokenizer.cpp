#include "Tokenizer.h"
#include "ThreadPool.h"
#include "Toolkit.h"
#include <thread>
#include <future>

Tokenizer::Tokenizer(const std::vector<std::string>& vocabList) : vocab(vocabList), idToToken(vocabList) {
    /*
    Input:
        - vocabList: A list of vocabulary strings.
    Output:
        - Constructs the Tokenizer object, mapping tokens to IDs and vice versa.
    Functionality:
        - Adds an "<UNK>" token if not present for handling unknown tokens.
    */

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

std::vector<int> Tokenizer::encode(const std::vector<std::string>& tokens, const std::string& logFile) {
    /*
    Input:
        - tokens: A vector of strings to encode.
        - logFile: A string specifying the name of the file to write the output to (default is "Outputs.txt", don't write if logFile = "").
    Output:
        - A vector of integers representing the IDs of the tokens.
    Functionality:
        - Maps each token to its corresponding ID. Unknown tokens are mapped to "<UNK>".
    */

    std::vector<int> encodedTokens;
    for (const auto& token : tokens) {
        if (tokenToId.find(token) != tokenToId.end()) {
            encodedTokens.push_back(tokenToId[token]);
        }
        else {
            encodedTokens.push_back(unknownId);
        }
    }

    writeToFile("Encode", encodedTokens, logFile);
    return encodedTokens;
}

std::vector<std::string> Tokenizer::decode(const std::vector<int>& ids, const std::string& logFile) {
    /*
    Input:
        - ids: A vector of token IDs to decode.
        - logFile: A string specifying the name of the file to write the output to (default is "Outputs.txt", don't write if logFile = "").
    Output:
        - A vector of strings representing the decoded tokens.
    Functionality:
        - Maps each ID to its corresponding token. Throws an exception for invalid IDs.
    */

    std::vector<std::string> decodedTokens;
    for (const auto& id : ids) {
        if (id >= 0 && id < static_cast<int>(idToToken.size())) {
            decodedTokens.push_back(idToToken[id]);
        }
        else {
            throw std::out_of_range("Invalid token ID in decode.");
        }
    }

    writeToFile("Decode", decodedTokens, logFile);
    return decodedTokens;
}

std::vector<std::vector<int>> Tokenizer::batchEncode(const std::vector<std::vector<std::string>>& sentences, int numThreads, const std::string& logFile) {
    /*
    Input:
        - sentences: A batch of token sequences.
        - numThreads: The number of threads to use for processing (default is 2 and -1 is get all).
        - logFile: A string specifying the name of the file to write the output to (default is "Outputs.txt", don't write if logFile = "").
    Output:
        - A vector of vectors, where each inner vector contains encoded token IDs for a sentence.
    Functionality:
        - Parallelizes the encoding process using multiple threads.
    */

    size_t numSentences = sentences.size();
    size_t maxThreads = std::thread::hardware_concurrency();

    if (numThreads <= 0 || numThreads > static_cast<int>(maxThreads)) {
        numThreads = maxThreads;
    }

    size_t blockSize = (numSentences + numThreads - 1) / numThreads;
    ThreadPool pool(numThreads);

    std::vector<std::future<std::vector<std::vector<int>>>> futures;

    for (int t = 0; t < numThreads; ++t) {
        size_t start = t * blockSize;
        size_t end = std::min(start + blockSize, numSentences);

        // Use the ThreadPool to enqueue tasks
        futures.push_back(pool.enqueue([this, &sentences, start, end]() {
            std::vector<std::vector<int>> blockResult;
            for (size_t i = start; i < end; ++i) {
                blockResult.push_back(this->encode(sentences[i],""));
            }
            return blockResult;
            }));
    }

    std::vector<std::vector<int>> results;
    for (auto& future : futures) {
        auto blockResult = future.get();
        results.insert(results.end(), blockResult.begin(), blockResult.end());
    }

    writeToFile("Batch Encode", results, logFile);
    return results;
}

std::vector<std::vector<std::string>> Tokenizer::batchDecode(const std::vector<std::vector<int>>& encodedSentences, int numThreads, const std::string& logFile) {
    /*
    Input:
        - encodedSentences: A batch of token ID sequences.
        - numThreads: The number of threads to use for processing (default is 2 and -1 is get all).
        - logFile: A string specifying the name of the file to write the output to (default is "Outputs.txt", don't write if logFile = "").
    Output:
        - A vector of vectors, where each inner vector contains decoded tokens for a sentence.
    Functionality:
        - Parallelizes the decoding process using multiple threads.
    */

    size_t numSentences = encodedSentences.size();
    size_t maxThreads = std::thread::hardware_concurrency();

    if (numThreads <= 0 || numThreads > static_cast<int>(maxThreads)) {
        numThreads = maxThreads;
    }

    size_t blockSize = (numSentences + numThreads - 1) / numThreads;
    ThreadPool pool(numThreads);

    std::vector<std::future<std::vector<std::vector<std::string>>>> futures;

    for (int t = 0; t < numThreads; ++t) {
        size_t start = t * blockSize;
        size_t end = std::min(start + blockSize, numSentences);

        // Use the ThreadPool to enqueue tasks
        futures.push_back(pool.enqueue([this, &encodedSentences, start, end]() {
            std::vector<std::vector<std::string>> blockResult;
            for (size_t i = start; i < end; ++i) {
                blockResult.push_back(this->decode(encodedSentences[i],""));
            }
            return blockResult;
            }));
    }

    std::vector<std::vector<std::string>> results;
    for (auto& future : futures) {
        auto blockResult = future.get();
        results.insert(results.end(), blockResult.begin(), blockResult.end());
    }

    writeToFile("Batch Decode", results, logFile);
    return results;
}

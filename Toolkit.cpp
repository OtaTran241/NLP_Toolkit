#include "Toolkit.h"
#include <sstream>
#include <algorithm>
#include <iostream>
#include <random>
#include <thread>
#include <future>
#include <mutex>
#include <regex>

std::vector<std::vector<std::string>> splitTokens(const std::vector<std::string>& tokens, size_t size) {
    /*
    Input:
        - tokens: A vector of strings (tokens) to be split.
        - size: The number of chunks to divide the tokens into.
    Output:
        - A vector of token chunks (vectors).
    Functionality:
        - Divides the tokens evenly among the specified number of threads.
    */

    size_t blockSize = tokens.size() / size;
    std::vector<std::vector<std::string>> splitBlocks;
    for (size_t i = 0; i < size; ++i) {
        size_t start = i * blockSize;
        size_t end = (i == size - 1) ? tokens.size() : (i + 1) * blockSize;
        splitBlocks.emplace_back(tokens.begin() + start, tokens.begin() + end);
    }
    return splitBlocks;
}

std::vector<std::string> Toolkit::tokenize(const std::string& text) {
    /*
    Input:
        - text: A string to be tokenized.
    Output:
        - A vector of tokens (words).
    Functionality:
        - Splits the input string into words using whitespace as the delimiter.
    */

    std::vector<std::string> tokens;
    std::istringstream stream(text);
    std::string word;

    while (stream >> word) {
        tokens.push_back(word);
    }
    return tokens;
}

std::unordered_map<std::string, int> Toolkit::getBagOfWords(const std::vector<std::string>& tokens, int numThreads) {
    /*
    Input:
        - tokens: A vector of strings (tokens).
        - numThreads: The number of threads to use for processing (default is 2 and -1 is get all).
    Output:
        - An unordered map where keys are words and values are their frequencies.
    Functionality:
        - Splits the tokens into chunks and processes each chunk in parallel to count token occurrences.
    */

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
    /*
    Input:
        - tokens: A vector of strings (tokens).
        - n: The desired n-gram size.
    Output:
        - A vector of n-grams, where each n-gram is a string formed by concatenating `n` consecutive tokens.
    Functionality:
        - Creates n-grams by grouping `n` consecutive tokens and joining them with a space.
        - Returns an empty vector if the input tokens are empty or if `n` is less than or equal to 0.
    */

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
    /*
    Input:
        - text: A string to be converted to lowercase.
    Output:
        - A new string where all uppercase letters are converted to lowercase.
    Functionality:
        - Uses `std::transform` to convert all characters in the input string to lowercase.
    */

    std::string result = text;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

std::string Toolkit::removePunctuation(const std::string& text) {
    /*
    Input:
        - text: A string from which punctuation will be removed.
    Output:
        - A new string with all punctuation characters removed.
    Functionality:
        - Iterates through the input string and appends non-punctuation characters to the result.
        - Uses `std::ispunct` to check for punctuation characters.
    */

    std::string result;
    for (char ch : text) {
        if (!std::ispunct(ch)) {
            result += ch;
        }
    }
    return result;
}

std::unordered_map<std::string, std::vector<float>> Toolkit::getEmbeddings(const std::vector<std::string>& tokens, size_t embeddingSize, int numThreads) {
    /*
    Input:
        - tokens: A vector of strings for which embeddings will be generated.
        - embeddingSize: The size of the embedding vector for each token.
        - numThreads: The number of threads to use for parallel processing (default is 2 and -1 is get all).
    Output:
        - An unordered map where keys are tokens and values are randomly generated embedding vectors.
    Functionality:
        - Splits the tokens into chunks and generates random embeddings for each token in parallel.
    */

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
    /*
    Input:
        - text: A word to be stemmed.
    Output:
        - The stemmed version of the input word.
    Functionality:
        - Applies basic stemming rules by removing common suffixes ("ing", "ed", "es", "s", "er").
        - Ensures the stemmed word is not too short by checking the minimum size.
        - Uses regex to identify and handle suffixes efficiently.
    */

    if (text.size() <= 3) {
        return text;
    }

    std::string stemmed = text;

    if (std::regex_match(text, std::regex(".*ing$"))) {
        if (text.size() > 4 && text[text.size() - 4] == text[text.size() - 5]) {
            stemmed = text.substr(0, text.size() - 4);
        }
        else {
            stemmed = text.substr(0, text.size() - 3);
        }
    }
    else if (std::regex_match(text, std::regex(".*ed$"))) {
        stemmed = text.substr(0, text.size() - 2);
    }
    else if (std::regex_match(text, std::regex(".*es$"))) {
        stemmed = text.substr(0, text.size() - 2);
    }
    else if (std::regex_match(text, std::regex(".*s$"))) {
        stemmed = text.substr(0, text.size() - 1);
    }
    else if (std::regex_match(text, std::regex(".*er$"))) {
        stemmed = text.substr(0, text.size() - 2);
    }

    if (stemmed.size() < 3) {
        return text;
    }

    return stemmed;
}

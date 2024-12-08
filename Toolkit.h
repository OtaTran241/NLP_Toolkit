#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <variant>
#include <iomanip>
#include <fstream>

using OutputType = std::variant<
    std::string,
    std::vector<int>,
    std::vector<std::string>,
    std::vector<std::vector<int>>,
    std::vector<std::vector<std::string>>,
    std::unordered_map<std::string, int>,
    std::unordered_map<std::string, std::vector<float>>
>;

void writeToFile(const std::string& taskName, const OutputType& output, const std::string& fileName = "Outputs.txt");

class Toolkit {
public:
    static std::vector<std::string> tokenize(const std::string& text, const std::string& logFile = "Outputs.txt");

    static std::unordered_map<std::string, int> getBagOfWords(const std::vector<std::string>& tokens, int numThreads = 2, const std::string& logFile = "Outputs.txt");

    static std::vector<std::string> getNGrams(const std::vector<std::string>& tokens, int n, const std::string& logFile = "Outputs.txt");

    static std::string toLower(const std::string& text, const std::string& logFile = "Outputs.txt");
    static std::string removePunctuation(const std::string& text, const std::string& logFile = "Outputs.txt");

    static std::unordered_map<std::string, std::vector<float>> getEmbeddings(const std::vector<std::string>& tokens, size_t vectorEmbeddingSize = 300, int numThreads = 2, const std::string& logFile = "Outputs.txt");

    static std::string stem(const std::string& text, const std::string& logFile = "Outputs.txt");

    static std::string removeSpecialCharacters(const std::string& text, const std::string& specialCharFile, int numThreads = 2, const std::string& logFile = "Outputs.txt");

    static std::string removeStopWords(const std::string& text, const std::string& stopWordsFile, int numThreads = 2, const std::string& logFile = "Outputs.txt");
};

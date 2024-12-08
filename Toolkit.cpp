#include "Toolkit.h"
#include "ThreadPool.h"
#include <sstream>
#include <algorithm>
#include <random>
#include <thread>
#include <future>
#include <mutex>
#include <regex>
#include <unordered_set>

void writeToFile(const std::string& taskName, const OutputType& output, const std::string& fileName) {
    /*
    Input:
        - taskName: A string representing the name of the task.
        - output: A variant (std::variant) containing various possible data types:
            - std::vector<int>
            - std::vector<std::vector<int>>
            - std::vector<std::string>
            - std::vector<std::vector<std::string>>
            - std::unordered_map<std::string, int>
            - std::unordered_map<std::string, std::vector<float>>
            - std::string
        - fileName: A string specifying the name of the file to write the output to (return if fileName = "").
    Output:
        - None (void). The function writes formatted data to the specified file.
    Functionality:
        - Opens the specified file in append mode. If the file cannot be opened, prints an error message.
        - Writes a header with the task name to the file.
        - Processes the `output` based on its type and writes its contents in a human-readable format to the file.
            - For vectors of integers or strings: Writes each element, separating them by spaces or newlines.
            - For nested vectors: Writes inner vectors on separate lines.
            - For unordered maps: Writes key-value pairs, with values formatted appropriately.
            - For strings: Writes the string directly.
        - Ensures the file is properly closed after writing.
    */
    if (fileName == "") {
        std::cout << "\033[33mSkip write task: " << taskName << "\033[0m\n";
        return;
    }

    std::ofstream outFile(fileName, std::ios::app);
    if (!outFile) {
        std::cout << "\033[31mFailed to open file: " << fileName << "\033[0m\n";
        return;
    }

    outFile << "======Task: " << taskName << "======\n";

    std::visit([&outFile](auto&& value) {
        using T = std::decay_t<decltype(value)>;

        if constexpr (std::is_same_v<T, std::vector<int>>) {
            for (const auto& item : value) {
                outFile << item << " ";
            }
            outFile << "\n";
        }
        else if constexpr (std::is_same_v<T, std::vector<std::vector<int>>>) {
            for (const auto& vec : value) {
                for (const auto& item : vec) {
                    outFile << item << " ";
                }
                outFile << "\n";
            }
        }
        else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
            for (const auto& item : value) {
                outFile << item << "\n";
            }
        }
        else if constexpr (std::is_same_v<T, std::vector<std::vector<std::string>>>) {
            for (const auto& vec : value) {
                for (const auto& item : vec) {
                    outFile << item << " ";
                }
                outFile << "\n";
            }
        }
        else if constexpr (std::is_same_v<T, std::unordered_map<std::string, int>>) {
            for (const auto& pair : value) {
                outFile << pair.first << ": " << pair.second << "\n";
            }
        }
        else if constexpr (std::is_same_v<T, std::unordered_map<std::string, std::vector<float>>>) {
            for (const auto& pair : value) {
                outFile << pair.first << ": ";
                for (const auto& num : pair.second) {
                    outFile << num << " ";
                }
                outFile << "\n";
            }
        }
        else if constexpr (std::is_same_v<T, std::string>) {
            outFile << value << "\n";
        }

        }, output);

    outFile.close();
}

std::vector<std::vector<std::string>> splitTokens(const std::vector<std::string>& tokens, size_t numParts) {
    /*
    Input:
        - tokens: A vector of strings (tokens) to be split.
        - numParts: The number of chunks to divide the tokens into.
    Output:
        - A vector of token chunks (vectors).
    Functionality:
        - Divides the tokens evenly among the specified number of threads.
    */

    std::vector<std::vector<std::string>> result(numParts);
    size_t blockSize = tokens.size() / numParts;
    size_t remainder = tokens.size() % numParts;

    for (size_t i = 0, start = 0; i < numParts; ++i) {
        size_t end = start + blockSize + (i < remainder ? 1 : 0);
        result[i] = std::vector<std::string>(tokens.begin() + start, tokens.begin() + end);
        start = end;
    }

    return result;
}

std::unordered_set<std::string> readFromFileTXT(const std::string& filename) {
    std::unordered_set<std::string> items;
    std::ifstream file(filename);
    std::string line;

    if (file.is_open()) {
        while (std::getline(file, line)) {
            items.insert(line);
        }
        file.close();
    }
    else {
        std::cout << "\n\033[31mFailed to open file: " << filename << "\033[0m\n";
        return items;
    }

    return items;
}

std::vector<std::string> Toolkit::tokenize(const std::string& text, const std::string& logFile) {
    /*
    Input:
        - text: A string to be tokenized.
        - logFile: A string specifying the name of the file to write the output to (default is "Outputs.txt", don't write if logFile = "").
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

    writeToFile("Tokenize", tokens, logFile);
    return tokens;
}

std::unordered_map<std::string, int> Toolkit::getBagOfWords(const std::vector<std::string>& tokens, int numThreads, const std::string& logFile) {
    /*
    Input:
        - tokens: A vector of strings (tokens).
        - numThreads: The number of threads to use for processing (default is 2 and -1 is get all).
        - logFile: A string specifying the name of the file to write the output to (default is "Outputs.txt", don't write if logFile = "").
    Output:
        - An unordered map where keys are words and values are their frequencies.
    Functionality:
        - Splits the tokens into chunks and processes each chunk in parallel to count token occurrences.
    */

    size_t maxThreads = std::thread::hardware_concurrency();
    if (numThreads <= 0 || numThreads > static_cast<int>(maxThreads)) {
        numThreads = maxThreads;
    }

    std::vector<std::unordered_map<std::string, int>> results(numThreads);
    auto splitBlocks = splitTokens(tokens, numThreads);

    ThreadPool pool(numThreads);

    for (int i = 0; i < numThreads; ++i) {
        pool.enqueue([&results, &splitBlocks, i] {
            for (const auto& token : splitBlocks[i]) {
                results[i][token]++;
            }
            });
    }

    pool.~ThreadPool();

    std::unordered_map<std::string, int> combinedResult;
    for (const auto& result : results) {
        for (const auto& [token, count] : result) {
            combinedResult[token] += count;
        }
    }

    writeToFile("Bag Of Words", combinedResult, logFile);
    return combinedResult;
}

std::vector<std::string> Toolkit::getNGrams(const std::vector<std::string>& tokens, int n, const std::string& logFile) {
    /*
    Input:
        - tokens: A vector of strings (tokens).
        - n: The desired n-gram size.
        - logFile: A string specifying the name of the file to write the output to (default is "Outputs.txt", don't write if logFile = "").
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

    std::string task = std::to_string(n) + "-Grams";
    writeToFile(task, ngrams, logFile);
    return ngrams;
}

std::string Toolkit::toLower(const std::string& text, const std::string& logFile) {
    /*
    Input:
        - text: A string to be converted to lowercase.
        - logFile: A string specifying the name of the file to write the output to (default is "Outputs.txt", don't write if logFile = "").
    Output:
        - A new string where all uppercase letters are converted to lowercase.
    Functionality:
        - Uses `std::transform` to convert all characters in the input string to lowercase.
    */

    std::string result = text;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);

    writeToFile("To Lower", result, logFile);
    return result;
}

std::string Toolkit::removePunctuation(const std::string& text, const std::string& logFile) {
    /*
    Input:
        - text: A string from which punctuation will be removed.
        - logFile: A string specifying the name of the file to write the output to (default is "Outputs.txt", don't write if logFile = "").
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

    writeToFile("Remove Punctuation", result, logFile);
    return result;
}

std::unordered_map<std::string, std::vector<float>> Toolkit::getEmbeddings(const std::vector<std::string>& tokens, size_t embeddingSize, int numThreads, const std::string& logFile) {
    /*
    Input:
        - tokens: A vector of strings for which embeddings will be generated.
        - embeddingSize: The size of the embedding vector for each token.
        - numThreads: The number of threads to use for parallel processing (default is 2 and -1 is get all).
        - logFile: A string specifying the name of the file to write the output to (default is "Outputs.txt", don't write if logFile = "").
    Output:
        - An unordered map where keys are tokens and values are randomly generated embedding vectors.
    Functionality:
        - Splits the tokens into chunks and generates random embeddings for each token in parallel.
    */

    size_t maxThreads = std::thread::hardware_concurrency();
    if (numThreads <= 0 || numThreads > static_cast<int>(maxThreads)) {
        numThreads = maxThreads;
    }

    std::vector<std::unordered_map<std::string, std::vector<float>>> results(numThreads);
    auto splitBlocks = splitTokens(tokens, numThreads);

    ThreadPool pool(numThreads);

    for (int i = 0; i < numThreads; ++i) {
        pool.enqueue([&results, &splitBlocks, i, embeddingSize] {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(-1.0, 1.0);

            for (const auto& token : splitBlocks[i]) {
                std::vector<float> embedding(embeddingSize);
                for (auto& value : embedding) {
                    value = static_cast<float>(dis(gen));
                }
                results[i][token] = embedding;
            }
            });
    }

    pool.~ThreadPool();

    std::unordered_map<std::string, std::vector<float>> combinedEmbeddings;
    for (const auto& result : results) {
        for (const auto& [token, embedding] : result) {
            combinedEmbeddings[token] = embedding;
        }
    }

    writeToFile("Embeddings", combinedEmbeddings, logFile);
    return combinedEmbeddings;
}

std::string Toolkit::stem(const std::string& text, const std::string& logFile) {
    /*
    Input:
        - text: A word to be stemmed.
        - logFile: A string specifying the name of the file to write the output to (default is "Outputs.txt", don't write if logFile = "").
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

    writeToFile("Stem", stemmed, logFile);
    return stemmed;
}

std::string Toolkit::removeSpecialCharacters(const std::string& text, const std::string& specialCharFile, int numThreads, const std::string& logFile) {
    /*
    Input:
        - text: A string to process.
        - specialCharFile: Path to a file containing special characters (one per line).
        - numThreads: Number of threads for parallel processing (default is 2 and -1 is get all).
        - logFile: A string specifying the name of the file to write the output to (default is "Outputs.txt", don't write if logFile = "").
    Output:
        - A string with special characters removed.
    */

    auto specialChars = readFromFileTXT(specialCharFile);

    size_t maxThreads = std::thread::hardware_concurrency();
    if (numThreads <= 0 || numThreads > static_cast<int>(maxThreads)) {
        numThreads = maxThreads;
    }

    std::vector<std::string> results(numThreads);
    std::vector<std::string> splitText(numThreads);

    size_t chunkSize = text.size() / numThreads;
    for (int i = 0; i < numThreads; ++i) {
        size_t start = i * chunkSize;
        size_t end = (i == numThreads - 1) ? text.size() : (i + 1) * chunkSize;
        splitText[i] = text.substr(start, end - start);
    }

    ThreadPool pool(numThreads);

    for (int i = 0; i < numThreads; ++i) {
        pool.enqueue([&splitText, &specialChars, &results, i] {
            std::string result;
            for (char ch : splitText[i]) {
                if (specialChars.find(std::string(1, ch)) == specialChars.end()) {
                    result += ch;
                }
            }
            results[i] = result;
            });
    }

    pool.~ThreadPool(); 

    std::string result;
    for (const auto& part : results) {
        result += part;
    }

    writeToFile("Remove Special Characters", result, logFile);
    return result;
}

std::string Toolkit::removeStopWords(const std::string& text, const std::string& stopWordsFile, int numThreads, const std::string& logFile) {
    /*
    Input:
        - text: A string to process.
        - stopWordsFile: Path to a file containing stop words (one per line).
        - numThreads: Number of threads for parallel processing (default is 2 and -1 is get all).
        - logFile: A string specifying the name of the file to write the output to (default is "Outputs.txt", don't write if logFile = "").
    Output:
        - A string with stop words removed.
    */

    auto stopWords = readFromFileTXT(stopWordsFile);
    auto tokens = tokenize(text);

    size_t maxThreads = std::thread::hardware_concurrency();
    if (numThreads <= 0 || numThreads > static_cast<int>(maxThreads)) {
        numThreads = maxThreads;
    }

    std::vector<std::vector<std::string>> results(numThreads);
    auto splitBlocks = splitTokens(tokens, numThreads);

    ThreadPool pool(numThreads);

    for (int i = 0; i < numThreads; ++i) {
        pool.enqueue([&results, &splitBlocks, &stopWords, i] {
            for (const auto& token : splitBlocks[i]) {
                if (stopWords.find(token) == stopWords.end()) {
                    results[i].push_back(token);
                }
            }
            });
    }

    pool.~ThreadPool(); 

    std::vector<std::string> filteredTokens;
    for (const auto& result : results) {
        filteredTokens.insert(filteredTokens.end(), result.begin(), result.end());
    }

    std::ostringstream result;
    for (size_t i = 0; i < filteredTokens.size(); ++i) {
        result << filteredTokens[i];
        if (i != filteredTokens.size() - 1) {
            result << " ";
        }
    }

    writeToFile("Remove Stop Words", result.str(), logFile);
    return result.str();
}
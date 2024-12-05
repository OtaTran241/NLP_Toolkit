#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <sstream>
#include <algorithm>
#define NOMINMAX
#include <windows.h> 
#include "Toolkit.h"
#include "Tokenizer.h" 

// Critical Section for print sync
CRITICAL_SECTION coutLock;

std::vector<std::string> tokens = { "hello", "world", "hello", "my", "name", "is", "My", "what", "is", "your", "name" };
std::vector<std::string> vocab = { "hello", "world", "<UNK>", "my", "name", "is", "<UNK>", "My" };
Tokenizer tokenizer(vocab);
std::string text = "Hello, world! This is a test for Tokenizer.";

void synchronizedPrint(const std::string& text) {
    EnterCriticalSection(&coutLock);
    std::cout << text << std::endl;
    LeaveCriticalSection(&coutLock);
}

// Test Toolkit
void testTokenize() {
    std::vector<std::string> tokens = Toolkit::tokenize(text);
    std::ostringstream oss;
    oss << "Tokens: ";
    for (const auto& token : tokens) oss << token << " ";
    oss << std::endl;
    synchronizedPrint(oss.str());
}

void testBagOfWords() {
    auto bag = Toolkit::getBagOfWords(tokens, 4);
    std::ostringstream oss;
    oss << "Bag of Words: ";
    for (const auto& [word, count] : bag) {
        oss << word << ": " << count << " ";
    }
    oss << std::endl;
    synchronizedPrint(oss.str());
}

void testNGrams() {
    int n = 2;
    auto ngrams = Toolkit::getNGrams(tokens, n);
    std::ostringstream oss;
    oss << n << "-grams: ";
    for (const auto& ngram : ngrams) oss << "\"" << ngram << "\" ";
    oss << std::endl;
    synchronizedPrint(oss.str());
}

void testNormalization() {
    std::string lower = Toolkit::toLower(text);
    std::string noPunctuation = Toolkit::removePunctuation(text);
    std::ostringstream oss;
    oss << "Lowercase: " << lower << "\nWithout Punctuation: " << noPunctuation;
    oss << std::endl;
    synchronizedPrint(oss.str());
}

void testEmbeddings() {
    auto embeddings = Toolkit::getEmbeddings(tokens, 3);
    std::ostringstream oss;
    oss << "Embeddings (showing first 5 values):" << std::endl;
    for (const auto& [token, vec] : embeddings) {
        oss << token << ": ";
        for (size_t i = 0; i < std::min(vec.size(), static_cast<size_t>(5)); ++i) {
            oss << vec[i] << " ";
        }
        oss << "..." << std::endl;
    }
    synchronizedPrint(oss.str());
}

void testStemming() {
    std::string text = "swimming";
    std::string stemmed = Toolkit::stem(text);
    synchronizedPrint("Stemmed: " + stemmed + "\n");
}

// Test Tokenizer
void testTokenizerEncode() {
    std::vector<std::string> sentence = { "hello", "unknown", "world", "is", "name" };

    auto encoded = tokenizer.encode(sentence);
    std::ostringstream oss;
    oss << "Tokenizer Encode: ";
    for (auto id : encoded) {
        oss << id << " ";
    }
    oss << std::endl;
    synchronizedPrint(oss.str());
}

void testTokenizerDecode() {
    std::vector<int> encoded = { 0, 2, 1, 5, 7, 3, 4 }; 

    auto decoded = tokenizer.decode(encoded);
    std::ostringstream oss;
    oss << "Tokenizer Decode: ";
    for (const auto& token : decoded) {
        oss << token << " ";
    }
    oss << std::endl;
    synchronizedPrint(oss.str());
}

void testTokenizerBatchEncode() {
    std::vector<std::vector<std::string>> sentences = {
        {"hello", "world", "test"},
        {"unknown", "hello", "name", "My"}
    };

    auto batchEncoded = tokenizer.batchEncode(sentences, 5);
    std::ostringstream oss;
    oss << "Tokenizer Batch Encode:" << std::endl;
    for (const auto& encoded : batchEncoded) {
        for (auto id : encoded) {
            oss << id << " ";
        }
        oss << std::endl;
    }
    synchronizedPrint(oss.str());
}

void testTokenizerBatchDecode() {
    std::vector<std::vector<int>> encodedSentences = {
        {0, 1, 2, 4, 3},
        {2, 0, 6, 5}
    };

    auto batchDecoded = tokenizer.batchDecode(encodedSentences, 3); 
    std::ostringstream oss;
    oss << "Tokenizer Batch Decode:" << std::endl;
    for (const auto& decoded : batchDecoded) {
        for (const auto& token : decoded) {
            oss << token << " ";
        }
        oss << std::endl;
    }
    synchronizedPrint(oss.str());
}

// Multi-thread testing
void testAllInParallel() {
    HANDLE processes[10];

    // Create threads for each test function
    processes[0] = CreateThread(nullptr, 0, [](LPVOID) -> DWORD { testTokenize(); return 0; }, nullptr, 0, nullptr);
    processes[1] = CreateThread(nullptr, 0, [](LPVOID) -> DWORD { testBagOfWords(); return 0; }, nullptr, 0, nullptr);
    processes[2] = CreateThread(nullptr, 0, [](LPVOID) -> DWORD { testNGrams(); return 0; }, nullptr, 0, nullptr);
    processes[3] = CreateThread(nullptr, 0, [](LPVOID) -> DWORD { testNormalization(); return 0; }, nullptr, 0, nullptr);
    processes[4] = CreateThread(nullptr, 0, [](LPVOID) -> DWORD { testEmbeddings(); return 0; }, nullptr, 0, nullptr);
    processes[5] = CreateThread(nullptr, 0, [](LPVOID) -> DWORD { testStemming(); return 0; }, nullptr, 0, nullptr);
    processes[6] = CreateThread(nullptr, 0, [](LPVOID) -> DWORD { testTokenizerEncode(); return 0; }, nullptr, 0, nullptr);
    processes[7] = CreateThread(nullptr, 0, [](LPVOID) -> DWORD { testTokenizerDecode(); return 0; }, nullptr, 0, nullptr);
    processes[8] = CreateThread(nullptr, 0, [](LPVOID) -> DWORD { testTokenizerBatchEncode(); return 0; }, nullptr, 0, nullptr);
    processes[9] = CreateThread(nullptr, 0, [](LPVOID) -> DWORD { testTokenizerBatchDecode(); return 0; }, nullptr, 0, nullptr);

    // Wait for all threads to complete
    WaitForMultipleObjects(10, processes, TRUE, INFINITE);

    for (auto process : processes) {
        CloseHandle(process);
    }
}

int main() {
    InitializeCriticalSection(&coutLock);

    std::cout << "Testing Toolkit and Tokenizer with multi-threading:" << std::endl;
    std::cout << std::string(30, '-') << std::endl;
    testAllInParallel();

    DeleteCriticalSection(&coutLock);

    return 0;
}

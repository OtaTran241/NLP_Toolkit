
# NLP Toolkit: A High-Performance C++ Library for Natural Language Processing  

---

## **Project overview**  
**NLP Toolkit** is a high-performance library written in C++ designed to address the needs of Natural Language Processing (NLP) applications. By leveraging advanced C++ techniques such as multi-threading, batch processing, this toolkit is capable of handling large-scale text data efficiently. The library provides core NLP functionalities such as tokenization, stemming, embeddings(for simulation purposes only), and more, and can be easily extended for real-world applications.

---

## Table of Contents
- [Project Overview](#Project-overview)
- [Key Features](#Key-Features)
- [Technical Highlights](#Technical-Highlights)
- [Code Examples](#Code-Examples)
- [Future Update](#Future-Update)
- [Contributors](#Contributors)

---

## **Key Features**  
- **Efficient Tokenization**: 
  - Tokenize text into words or subwords with support for batch processing and multi-threading.

- **Bag-of-Words Construction**: 
  - Generates frequency counts of words in a given dataset using multi-threading for faster computation.

- **N-Gram Support**: 
  - Extract N-grams from text data to support feature extraction for NLP models.

- **Text Normalization**: 
  - Convert text to lowercase and remove punctuation efficiently.

- **Custom Word Embeddings**: 
  - Generate random embeddings with configurable dimensionality for tokens in text.

- **Stemming**: 
  - Extract the base form of a word, helping reduce vocabulary size in NLP tasks.

- **Dictionary-Based Encoding**: 
  - Provides an efficient `Tokenizer` class for encoding and decoding text into/from IDs, with robust handling of unknown words (`<UNK>`).
 
- **Demo of Multi-Process**:
  - In the `main` function, we demonstrate how various functionalities of the toolkit are tested in parallel using multi-threading. The multi-threading is used to run tests on functions like tokenization, bag-of-words, n-grams, normalization, embeddings, and stemming in parallel, without blocking the execution.

Here is an example:

```cpp
// Multi-process testing
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
```

```cpp
int main() {
    InitializeCriticalSection(&coutLock);

    std::cout << "Testing Toolkit with multi-threading:" << std::endl;
    std::cout << std::string(30, '-') << std::endl;
    testAllInParallel();

    DeleteCriticalSection(&coutLock);

    return 0;
}
```
---

## **Technical Highlights**  

1. **Multi-Threaded Processing**  
   - Utilizes C++ standard libraries for concurrent execution of tasks.
   - Optimized for multi-core, enabling faster processing of large text datasets.

2. **Batch Processing for Scalability**  
   - Both `Tokenizer` and `Bag-of-Words` implementations are optimized to process batches of text data in parallel.

3. **Custom Embeddings**  
   - Generates random embeddings with configurable dimensions to simulate vector space models.  

4. **Modern C++ Design**  
   - Written in C++17, employing modern constructs like `std::async`, `std::unordered_map`, and lambdas for clean, maintainable, and high-performance code.  

5. **Critical Section Synchronization**  
   - Uses critical sections for thread-safe logging, ensuring consistent and reliable output even in multi-threaded environments.  

---

## **Use Cases**  
- **Text Preprocessing**: Normalize, tokenize, and clean text data for downstream NLP tasks.  
- **Feature Extraction**: Create Bag-of-Words or N-gram representations for machine learning models.  
- **Custom NLP Pipelines**: Use embeddings and stemming to build custom NLP solutions.  
- **Scalable Encoding**: Efficiently encode large datasets into ID sequences for deep learning models.

---

## **Quick Start**  

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/OtaTran241/NLP_Toolkit.git
   cd NLP_Toolkit
   ```
   **If you are using Visual Studio, you can simply run the 'NLP_Toolkit.sln' file and skip the other steps.**
   
3. **Build the Library**  
   Make sure you have a C++17-compatible compiler installed. Then, run:
   ```bash
   mkdir build && cd build
   cmake ..
   make
   ```

4. **Run the Tests**  
   After building, test the library using the provided test cases:
   ```bash
   ./test_NLPToolkit
   ```

5. **Integrate with Your Project**  
   Include the library in your C++ project by linking the compiled binaries and including the headers.

---

## **Code Examples**  
**You can either use or modify the current 'main.cpp' with existing multi-processing (just update the input) or organize a new 'main.cpp' file.**
Here is an example: 
```cpp
#include "Toolkit.h"
#include "Tokenizer.h"

int main() {
    // Input text
    std::string text = "Hello, world! Welcome to NLP Toolkit.";
    std::vector<std::string> batch = {"Hello world", "Tokenization is fast"};

    // Tokenization
    auto tokens = Toolkit::tokenize(text);
    auto ngrams = Toolkit::getNGrams(tokens, 2);

    // Normalization
    std::string lowerText = Toolkit::toLower(text);

    // Encoding & Decoding
    Tokenizer tokenizer({"<UNK>", "hello", "world"});
    auto encoded = tokenizer.encode(batch);
    auto decoded = tokenizer.decode(encoded);

    return 0;
}
```

---

## **Future Update**  
- **Integration with Python (can Pip install)**: Provide Python bindings using Pybind11 for seamless use in Python-based NLP pipelines.
- **Input file**: Supports input in various formats such as PDF, CSV, API,...
- **Advanced Features**: Support for lemmatization, named entity recognition, and dependency parsing.  

---  
## **Contributors**  
Contributions are welcome! If you have any ideas for improving the model or adding new features, feel free to submit a pull request or send an email to tranducthuan220401@gmail.com.

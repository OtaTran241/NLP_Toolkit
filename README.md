
# NLP Toolkit: A High-Performance C++ Library for Natural Language Processing  

---

## **Overview**  
**NLP Toolkit** is a high-performance library written in C++ designed to address the needs of Natural Language Processing (NLP) applications. By leveraging advanced C++ techniques such as multi-threading, batch processing, and GPU acceleration, this toolkit is capable of handling large-scale text data efficiently. The library provides core NLP functionalities such as tokenization, stemming, embeddings, and more, and can be easily extended for real-world applications.

---

## **Key Features**  
- **Efficient Tokenization**: 
  - Tokenize text into words or subwords with support for batch processing and multi-threading.
  - GPU acceleration ensures performance at scale for large datasets.

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

---

## **Technical Highlights**  

1. **Multi-Threaded Processing**  
   - Utilizes C++ standard libraries for concurrent execution of tasks.
   - Optimized for multi-core CPUs, enabling faster processing of large text datasets.

2. **Batch Processing for Scalability**  
   - Both `Tokenizer` and `Bag-of-Words` implementations are optimized to process batches of text data in parallel.

3. **GPU Acceleration**  
   - Integrated with GPU computation for tasks such as tokenization and embedding generation, where supported.

4. **Custom Embeddings**  
   - Generates random embeddings with configurable dimensions to simulate vector space models.  

5. **Modern C++ Design**  
   - Written in C++17, employing modern constructs like `std::async`, `std::unordered_map`, and lambdas for clean, maintainable, and high-performance code.  

6. **Critical Section Synchronization**  
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

2. **Build the Library**  
   Make sure you have a C++17-compatible compiler installed. Then, run:
   ```bash
   mkdir build && cd build
   cmake ..
   make
   ```

3. **Run the Tests**  
   After building, test the library using the provided test cases:
   ```bash
   ./test_NLPToolkit
   ```

4. **Integrate with Your Project**  
   Include the library in your C++ project by linking the compiled binaries and including the headers.

---

## **Code Examples**  

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

## **Performance Benchmarks**  

| **Feature**             | **Dataset Size** | **Execution Time** | **Remarks**                 |
|--------------------------|------------------|---------------------|-----------------------------|
| Tokenization             | 1M words         | 0.5 seconds         | Multi-threaded processing   |
| Bag-of-Words             | 1M tokens        | 1.2 seconds         | Multi-core optimization     |
| Embedding Generation     | 100K words       | 0.8 seconds         | GPU acceleration            |
| Tokenizer Encode/Decode  | 100K sentences   | 0.6 seconds         | Batch processing enabled    |

---

## **Future Plans**  
- **Integration with Python**: Provide Python bindings using Pybind11 for seamless use in Python-based NLP pipelines.  
- **Advanced Features**: Support for lemmatization, named entity recognition, and dependency parsing.  
- **Deep Learning Compatibility**: Direct integration with PyTorch and TensorFlow.  

---  
## **Contributors**  
Developed by **OtaTran** with contributions from the open-source community. Contributions, bug reports, and feature suggestions are welcome!  

## **License**  
This project is licensed under the MIT License.  

---  
For more details, check out the [GitHub Repository](https://github.com/OtaTran241/NLP_Toolkit).  


# NLP Toolkit: Library for Natural Language Processing  

---

## **Project overview**  
**NLP Toolkit** is a high-performance library written in C++ designed to address the needs of Natural Language Processing (NLP) applications. By leveraging advanced C++ techniques such as multi-threading, batch processing, this toolkit is capable of handling large-scale text data efficiently. The library provides core NLP functionalities such as tokenization, stemming, embeddings(for simulation purposes only), and more, and can be easily extended for real-world applications.
### **Use Cases**  
- **Text Preprocessing**: Normalize, tokenize, and clean text data for downstream NLP tasks.  
- **Feature Extraction**: Create Bag-of-Words or N-gram representations for machine learning models.  
- **Custom NLP Pipelines**: Use embeddings and stemming to build custom NLP solutions.  
- **Scalable Encoding**: Efficiently encode large datasets into ID sequences for deep learning models.

---

## Table of Contents
- [Project Overview](#Project-overview)
- [Technical Highlights](#Technical-Highlights)
- [Key Features](#Key-Features)
- [Code Examples](#Code-Examples)
- [Future Update](#Future-Update)
- [Contributors](#Contributors)

---

## **Technical Highlights**  

1. **Multi-Threaded Processing**  
   - Utilizes C++ standard libraries for concurrent execution of tasks.
   - Optimized for multi-core, enabling faster processing of large text datasets.
  
2. **Thread Pooling**
  - The `ThreadPool` class is allows you to manage a collection of worker threads that can execute tasks concurrently. The pool of threads can handle multiple tasks by dispatching them to threads from the pool, which helps avoid the overhead of frequently creating and destroying threads.
    - **Concurrent task execution**: Tasks are distributed across multiple worker threads, making it easy to execute multiple operations concurrently.
    - **Task queuing**: Tasks can be enqueued and will be executed as soon as a worker thread is available.
    - **Graceful shutdown**: The thread pool can be stopped cleanly, ensuring all pending tasks are completed before termination.
    - **Return values**: Supports tasks that return values, using `std::future` to obtain the result of a task when it completes.
      
Here is an example to create a `ThreadPool`:
```cpp
ThreadPool pool(4); // Creates a pool with 4 threads
```

3. **Batch Processing for Scalability**  
   - Both class `Tokenizer`and `Toolkit` implementations are optimized to process batches of text data in parallel.

4. **Custom Embeddings**  
   - Generates random embeddings with configurable dimensions to simulate vector space models.  

5. **Modern C++ Design**  
   - Written in C++17, employing modern constructs like `std::async`, `std::unordered_map`, and lambdas for clean, maintainable, and high-performance code.  

6. **Critical Section Synchronization**  
   - Uses critical sections for thread-safe logging, ensuring consistent and reliable output even in multi-threaded environments.  

---

## **Key Features**  
**All outputs of the tasks will be saved to a .txt file (default is Outputs.txt and "" to not write outputs to a .txt file). You can modify the file path if desired.**  
Here is an example:
```cpp
// Change output file path to "MyOutputs/Output_BagOfWords.txt"
auto bag = Toolkit::getBagOfWords(tokens, 4, "MyOutputs/Output_BagOfWords.txt");

// Change output file path to "MyOutputs/Output_BatchEncode.txt"
auto batchEncoded = tokenizer.batchEncode(sentences, 5, "MyOutputs/Output_BatchEncode.txt");

// Do not write outputs to txt
auto decoded = tokenizer.decode(encoded, "");
```
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
    
- **Remove Special Characters or Remove Stop Words**:
  - Remove special characters (special_characters.txt) or remove stop words (stop_words.txt) from text data, you can update two this .txt files to customize them.
    
- **Dictionary-Based Encoding**: 
  - Provides an efficient `Tokenizer` class for encoding and decoding text into/from IDs, with robust handling of unknown words (`<UNK>`).

---

## **Quick Start**  

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/OtaTran241/NLP_Toolkit.git
   cd NLP_Toolkit
   ```
   **If you are using Visual Studio, you can simply run the 'NLP_Toolkit.sln' file and skip the other steps. (Window only)**
   
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
**You can either use or modify the current 'main.cpp' with existing multi-processing just update the input (Window only) or organize a new 'main.cpp' file.**  
Here is an example: 
```cpp
#include "Toolkit.h"
#include "Tokenizer.h"

int main() {
    // Input text
    std::string text = "Hello, world! Welcome to NLP Toolkit.";
    std::vector<std::string> batch = {"Hello world", "Tokenization is fast"};

    // Tokenization
    auto tokens = Toolkit::tokenize(text, "tokenize_output.txt");
    auto ngrams = Toolkit::getNGrams(tokens, 2, "2Grams_output.txt");

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
- **GPU Support**: Enables faster and more efficient processing through CUDA and GPU.   

---  
## **Contributors**  
Contributions are welcome! If you have any ideas for improving the model or adding new features, feel free to submit a pull request or send an email to tranducthuan220401@gmail.com.

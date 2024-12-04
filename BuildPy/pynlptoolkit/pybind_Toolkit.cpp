#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <unordered_map>
#include <vector>
#include <string>
#include "Toolkit.h"
#include "Tokenizer.h"

namespace py = pybind11;

// Bind Toolkit methods
void bindToolkit(py::module_& m) {
    py::class_<Toolkit>(m, "Toolkit")
        .def_static("tokenize", &Toolkit::tokenize, py::arg("text"),
            "Tokenize a string into words")
        .def_static("toLower", &Toolkit::toLower, py::arg("text"),
            "Convert string to lowercase")
        .def_static("removePunctuation", &Toolkit::removePunctuation, py::arg("text"),
            "Remove punctuation from a string")
        .def_static("getBagOfWords", &Toolkit::getBagOfWords, py::arg("tokens"), py::arg("numThreads") = 2,
            "Generate bag of words from tokens")
        .def_static("getNGrams", &Toolkit::getNGrams, py::arg("tokens"), py::arg("n"),
            "Generate n-grams from tokens")
        .def_static("stem", &Toolkit::stem, py::arg("word"),
            "Stem a word")
        .def_static("getEmbeddings", &Toolkit::getEmbeddings, py::arg("tokens"), py::arg("embeddingSize") = 100, py::arg("numThreads") = 2,
            "Generate random embeddings for tokens");
}

// Bind Tokenizer methods
void bindTokenizer(py::module_& m) {
    py::class_<Tokenizer>(m, "Tokenizer")
        .def(py::init<const std::vector<std::string>&>(), py::arg("vocab"),
            "Initialize a Tokenizer with a vocabulary")
        .def("encode", &Tokenizer::encode, py::arg("tokens"),
            "Encode a list of tokens into their corresponding IDs")
        .def("decode", &Tokenizer::decode, py::arg("ids"),
            "Decode a list of IDs into their corresponding tokens")
        .def("batchEncode", &Tokenizer::batchEncode, py::arg("sentences"), py::arg("numThreads") = 2,
            "Encode a batch of sentences using multiple threads")
        .def("batchDecode", &Tokenizer::batchDecode, py::arg("encodedSentences"), py::arg("numThreads") = 2,
            "Decode a batch of encoded sentences using multiple threads");
}

// Pybind11 module definition
PYBIND11_MODULE(pybind_Toolkit, m) {
    m.doc() = "Pybind11 wrapper for Toolkit and Tokenizer";

    // Bind Toolkit and Tokenizer classes
    bindToolkit(m);
    bindTokenizer(m);
}

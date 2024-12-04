from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os

ext_modules = [
    Pybind11Extension(
        "pynlptoolkit", 
        [
            "pynlptoolkit/pybind_Toolkit.cpp",  
            "pynlptoolkit/Toolkit.cpp",      
            "pynlptoolkit/Tokenizer.cpp",     
        ],
        include_dirs=["pynlptoolkit"],  
        cxx_std=17, 
    ),
]

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pynlptoolkit",  
    version="0.1.0",
    author="OtaTran",  
    author_email="tranducthuan220401@example.com", 
    description="A Python NLP toolkit powered by C++ for text processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/OtaTran241/NLP_Toolkit", 
    license="TDT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "License :: OSI Approved :: TDT License",
        "Operating System :: OS Independent",
    ],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},  
    python_requires=">=3.6",  
)

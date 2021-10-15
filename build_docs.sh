#!/bin/bash
cd docs
echo "Rebuilding References"
rm -v ./source/reference/algorithms/*
poetry run sphinx-apidoc -o ./source/reference/algorithms/ ./../gnss_lib_py/algorithms/
rm -v ./source/reference/core/*
poetry run sphinx-apidoc -o ./source/reference/core/ ./../gnss_lib_py/core/
rm -v ./source/reference/io/*
poetry run sphinx-apidoc -o ./source/reference/io/ ./../gnss_lib_py/io/
rm -v ./source/reference/utils/*
poetry run sphinx-apidoc -o ./source/reference/utils/ ./../gnss_lib_py/utils/
echo "Cleaning up existing make"
poetry run make clean
echo "Building docs in html"
poetry run make html
cd ..
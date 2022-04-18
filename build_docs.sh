#!/bin/bash
poetry export -f requirements.txt --output ./docs/source/requirements.txt
cd docs
echo "Rebuilding References"
rm -rv ./source/reference/algorithms/*
poetry run sphinx-apidoc -f -M -o ./source/reference/algorithms/ ./../gnss_lib_py/algorithms/
rm -rv ./source/reference/core/*
poetry run sphinx-apidoc -f -M -o ./source/reference/core/ ./../gnss_lib_py/core/
rm -rv ./source/reference/parsers/*
poetry run sphinx-apidoc -f -M -o ./source/reference/parsers/ ./../gnss_lib_py/parsers/
rm -rv ./source/reference/utils/*
poetry run sphinx-apidoc -f -M -o ./source/reference/utils/ ./../gnss_lib_py/utils/
echo "Cleaning up existing make"
poetry run make clean
echo "Building docs in html"
poetry run make html


#!/bin/bash
# export requirements.txt for buildings docs
poetry export -f requirements.txt --output ./docs/source/requirements.txt --with dev --without-hashes
cd docs

echo "Rebuilding References"
rm -r ./source/reference/algorithms/*
poetry run sphinx-apidoc -f -M -q -o ./source/reference/algorithms/ ./../gnss_lib_py/algorithms/
rm -r ./source/reference/parsers/*
poetry run sphinx-apidoc -f -M -q -o ./source/reference/parsers/ ./../gnss_lib_py/parsers/
rm -r ./source/reference/utils/*
poetry run sphinx-apidoc -f -M -q -o ./source/reference/utils/ ./../gnss_lib_py/utils/
rm -r ./source/reference/test_algorithms/*
poetry run sphinx-apidoc -f -M -q -o ./source/reference/test_algorithms/ ./../tests/algorithms/
rm -r ./source/reference/test_parsers/*
poetry run sphinx-apidoc -f -M -q -o ./source/reference/test_parsers/ ./../tests/parsers/
rm -r ./source/reference/test_utils/*
poetry run sphinx-apidoc -f -M -q -o ./source/reference/test_utils/ ./../tests/utils/

# remove previously downloaded .csv files if they exist
rm ./source/*/*.csv

echo "Cleaning existing make"
poetry run make clean

echo "Building docs in html"
poetry run make html

# export requirements.txt for setup.py
cd ..
poetry export -f requirements.txt --output ./requirements.txt --without-hashes

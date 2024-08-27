#!/bin/bash
# export requirements.txt for buildings docs
poetry config warnings.export false
poetry export -f requirements.txt --output ./docs/source/requirements.txt --with dev --without-hashes
# export requirements.txt for Conda environment setup
poetry export -f requirements.txt --output ./requirements.txt --without-hashes
cd docs

echo "Rebuilding References"
rm -r ./source/reference/algorithms/*
poetry run sphinx-apidoc -f -M -q -o ./source/reference/algorithms/ ./../gnss_lib_py/algorithms/
rm -r ./source/reference/navdata/*
poetry run sphinx-apidoc -f -M -q -o ./source/reference/navdata/ ./../gnss_lib_py/navdata/
rm -r ./source/reference/parsers/*
poetry run sphinx-apidoc -f -M -q -o ./source/reference/parsers/ ./../gnss_lib_py/parsers/
rm -r ./source/reference/utils/*
poetry run sphinx-apidoc -f -M -q -o ./source/reference/utils/ ./../gnss_lib_py/utils/
rm -r ./source/reference/visualizations/*
poetry run sphinx-apidoc -f -M -q -o ./source/reference/visualizations/ ./../gnss_lib_py/visualizations/
rm -r ./source/reference/test_algorithms/*
poetry run sphinx-apidoc -f -M -q -o ./source/reference/test_algorithms/ ./../tests/algorithms/
rm -r ./source/reference/test_navdata/*
poetry run sphinx-apidoc -f -M -q -o ./source/reference/test_navdata/ ./../tests/navdata/
rm -r ./source/reference/test_parsers/*
poetry run sphinx-apidoc -f -M -q -o ./source/reference/test_parsers/ ./../tests/parsers/
rm -r ./source/reference/test_utils/*
poetry run sphinx-apidoc -f -M -q -o ./source/reference/test_utils/ ./../tests/utils/
rm -r ./source/reference/test_visualizations/*
poetry run sphinx-apidoc -f -M -q -o ./source/reference/test_visualizations/ ./../tests/visualizations/

# remove previously downloaded .csv files if they exist
rm ./source/*/*.csv

echo "Cleaning existing make"
poetry run make clean

echo "Building docs in html"
poetry run make html

cd ..

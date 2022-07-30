#!/bin/bash
poetry export -f requirements.txt --output ./docs/source/requirements.txt --dev
cd docs
echo "Rebuilding References"
rm -rv ./source/reference/algorithms/*
poetry run sphinx-apidoc -f -M -o ./source/reference/algorithms/ ./../gnss_lib_py/algorithms/
rm -rv ./source/reference/parsers/*
poetry run sphinx-apidoc -f -M -o ./source/reference/parsers/ ./../gnss_lib_py/parsers/
rm -rv ./source/reference/utils/*
poetry run sphinx-apidoc -f -M -o ./source/reference/utils/ ./../gnss_lib_py/utils/
rm -rv ./source/reference/test_algorithms/*
poetry run sphinx-apidoc -f -M -o ./source/reference/test_algorithms/ ./../tests/algorithms/
rm -rv ./source/reference/test_parsers/*
poetry run sphinx-apidoc -f -M -o ./source/reference/test_parsers/ ./../tests/parsers/
rm -rv ./source/reference/test_utils/*
poetry run sphinx-apidoc -f -M -o ./source/reference/test_utils/ ./../tests/utils/
echo "Cleaning up existing make"
poetry run make clean
echo "Building docs in html"
poetry run make html
cd ..
poetry export -f requirements.txt --output ./requirements.txt --without-hashes

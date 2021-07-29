rm -v ./source/reference/algorithms/*
sphinx-apidoc -o ./source/reference/algorithms/ ./../gnss-lib-py/algorithms/
rm -v ./source/reference/core/*
sphinx-apidoc -o ./source/reference/core/ ./../gnss-lib-py/core/
rm -v ./source/reference/io/*
sphinx-apidoc -o ./source/reference/io/ ./../gnss-lib-py/io/
rm -v ./source/reference/utils/*
sphinx-apidoc -o ./source/reference/utils/ ./../gnss-lib-py/utils/

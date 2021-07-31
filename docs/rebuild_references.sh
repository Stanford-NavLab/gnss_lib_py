rm -v ./source/reference/algorithms/*
sphinx-apidoc -o ./source/reference/algorithms/ ./../gnss_lib_py/algorithms/
rm -v ./source/reference/core/*
sphinx-apidoc -o ./source/reference/core/ ./../gnss_lib_py/core/
rm -v ./source/reference/io/*
sphinx-apidoc -o ./source/reference/io/ ./../gnss_lib_py/io/
rm -v ./source/reference/utils/*
sphinx-apidoc -o ./source/reference/utils/ ./../gnss_lib_py/utils/

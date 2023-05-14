# https://makefiletutorial.com/

.PHONY: all help run deps

echo: 
	echo "Makefile working"

## Runs the code
run:
	PYOPENCL_CTX="" python src/multigrid.py assets/dotted-pattern-picture-horse-rider-vector-illustration.jpg

run2:
	PYOPENCL_CTX="" python src/multigrid.py assets/vector-symbol-bike-silhouette-dotted-outline-illustration-line-art-style.jpg

## Install dependencies
deps:
	pip install -r requirements.txt



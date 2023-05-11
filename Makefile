# https://makefiletutorial.com/

.PHONY: all help run deps

echo: 
	echo "Makefile working"

## Runs the code
run:
	PYOPENCL_CTX="" python src/multigrid.py assets/dotted-pattern-picture-horse-rider-vector-illustration.jpg

## Install dependencies
deps:
	pip install -r requirements.txt



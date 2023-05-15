# https://makefiletutorial.com/

.PHONY: all help run deps

echo: 
	echo "Makefile working"

## Runs the code
run:
	PYOPENCL_CTX="" python src/multigrid.py assets/horse-rider.jpg

run2:
	PYOPENCL_CTX="" python src/multigrid.py assets/eiffel-tower.jpg

run3:
	PYOPENCL_CTX="" python src/multigrid.py assets/bike.jpg

run4:
	PYOPENCL_CTX="" python src/multigrid.py assets/world-map.jpg

## Install dependencies
deps:
	pip install -r requirements.txt



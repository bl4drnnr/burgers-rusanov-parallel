# Makefile for Burgers Equation Solver
# Convenience commands for running simulations

.PHONY: help install demo sequential parallel visualize performance clean

help:
	@echo "Burgers Equation Solver - Available targets:"
	@echo ""
	@echo "  make install       - Install dependencies"
	@echo "  make demo          - Run complete demonstration"
	@echo "  make sequential    - Run sequential solver"
	@echo "  make parallel      - Run parallel solver (4 procs)"
	@echo "  make visualize     - Visualize results"
	@echo "  make performance   - Run performance scaling study"
	@echo "  make clean         - Remove generated files"
	@echo ""

install:
	pip install -e .

demo:
	python3 main.py demo

sequential:
	python3 main.py sequential --nx 200 --t-final 0.5

parallel:
	python3 main.py parallel --procs 4 --nx 400 --t-final 0.5

parallel-8:
	python3 main.py parallel --procs 8 --nx 800 --t-final 0.5

visualize:
	@if [ -f results_parallel.npz ]; then \
		python3 main.py visualize results_parallel.npz --viz-type shock; \
	else \
		echo "Error: results_parallel.npz not found. Run 'make parallel' first."; \
	fi

performance:
	python3 main.py performance --run-study --max-procs 8 --nx 1000

performance-large:
	python3 main.py performance --run-study --max-procs 16 --nx 2000

test-sequential:
	python3 1_sequential_rusanov.py --nx 100 --t-final 0.1 --save test_seq.npz

test-parallel:
	mpiexec -n 2 python3 2_parallel_rusanov.py --nx 100 --t-final 0.1 --save test_par.npz

clean:
	rm -f *.npz
	rm -f *.png
	rm -f *.json
	rm -rf frames/
	rm -rf scaling_results/
	rm -rf __pycache__/
	rm -rf *.egg-info/
	find . -name "*.pyc" -delete
	find . -name ".DS_Store" -delete

cleanall: clean
	rm -rf .venv/

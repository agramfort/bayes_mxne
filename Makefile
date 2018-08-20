# simple makefile to simplify repetetive build env management tasks under posix

PYTHON ?= python
CYTHON ?= cython
PYTEST ?= pytest
CTAGS ?= ctags

all: clean test

clean-pyc:
	find . -name "*.pyc" | xargs rm -f
	find . -name "__pycache__" | xargs rm -rf

clean-so:
	find . -name "*.so" | xargs rm -f
	find . -name "*.pyd" | xargs rm -f

clean-build:
	rm -rf build

clean-ctags:
	rm -f tags

clean: clean-build clean-pyc clean-so clean-ctags

in: inplace # just a shortcut
inplace:
	$(PYTHON) setup.py build_ext -i

test-code:
	$(PYTEST) -v bayes_mxne

test-doc:
	$(PYTEST) $(shell find doc -name '*.rst' | sort)

test-coverage:
	rm -rf coverage .coverage
	$(PYTEST) bayes_mxne --showlocals -v --cov=bayes_mxne --cov-report=html:coverage

test-manifest:
	check-manifest --ignore doc;

test-docstyle:
	@echo "Running pydocstyle"
	@pydocstyle bayes_mxne

test: test-code test-doc test-manifest test-docstyle

trailing-spaces:
	find . -name "*.py" | xargs perl -pi -e 's/[ \t]*$$//'

cython:
	find -name "*.pyx" | xargs $(CYTHON)

ctags:
	# make tags for symbol based navigation in emacs and vim
	# Install with: sudo apt-get install exuberant-ctags
	$(CTAGS) -R *

.PHONY : doc-plot
doc-plot:
	make -C doc html

.PHONY : doc
doc:
	make -C doc html-noplot

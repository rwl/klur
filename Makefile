VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
MATURIN = $(VENV)/bin/maturin

# https://www.maturin.rs/#usage
dev:
	$(MATURIN) develop

build:
	$(MATURIN) build

install:
	$(PIP) install .

bootstrap:
	python3 -m venv .venv
	$(PIP) install numpy scipy maturin

check:
	$(PYTHON) simple.py

.PHONY: dev build bootstrap
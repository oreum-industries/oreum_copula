# Makefile
# NOTE:
# + Intended for install on MacOS Apple Silicon arm64 using Accelerate
#   (NOT Intel x86 using MKL via Rosetta 2)
# + Uses sh by default: to confirm shell create a recipe with $(info $(SHELL))
.PHONY: dev dev-js lint help install-env install-mamba slides test-dev-env\
		tlmgr uninstall-env uninstall-mamba
.SILENT: dev dev-js lint help install-env install-mamba slides test-dev-env\
		tlmgr uninstall-env uninstall-mamba
MAMBADL := https://github.com/conda-forge/miniforge/releases/download/24.11.3-0
MAMBAV := Miniforge3-MacOSX-arm64.sh
MAMBARCMSG := Please create file $(MAMBARC), importantly set `platform: osx-arm64`
MAMBARC := $(HOME)/.mambarc
MAMBADIR := $(HOME)/miniforge
PYTHON_DEFAULT = $(or $(shell which python3), $(shell which python))
PYTHON_ENV = $(MAMBADIR)/envs/oreum_copula/bin/python
ifneq ("$(wildcard $(PYTHON_ENV))","")
    PYTHON = $(PYTHON_ENV)
else
    PYTHON = $(PYTHON_DEFAULT)
endif

dev:  # create env for local dev
	make install-env
	export PATH=$(MAMBADIR)/envs/oreum_copula/bin:$$PATH; \
		export MAMBA_EXE='$(MAMBADIR)/bin/mamba'; \
		export MAMBA_ROOT_PREFIX='$(MAMBADIR)'; \
		$(PYTHON_ENV) -m pip index versions oreum_core; \
		$(PYTHON_ENV) -m pip install ".[dev,oreum_core_pypi]"; \
		$(PYTHON_ENV) -c "import numpy as np; np.__config__.show()" > dev/install_log/blas_info.txt; \
		pipdeptree -a > dev/install_log/pipdeptree.txt; \
		pipdeptree -a -r > dev/install_log/pipdeptree_rev.txt; \
		pip-licenses -saud -f markdown -i csv2md --output-file LICENSES_THIRD_PARTY.md; \
		pre-commit install; \
		pre-commit autoupdate;

dev-js:  # create env for local dev alongside latest live oreum_core on JS machine
	make install-env
	export PATH=$(MAMBADIR)/envs/oreum_copula/bin:$$PATH; \
		export MAMBA_EXE='$(MAMBADIR)/bin/mamba'; \
		export MAMBA_ROOT_PREFIX='$(MAMBADIR)'; \
		$(PYTHON_ENV) -m pip install ".[dev]"; \
		$(PYTHON_ENV) -m pip install -e "../oreum_core[pymc]"; \
		$(PYTHON_ENV) -c "import numpy as np; np.__config__.show()" > dev/install_log/blas_info.txt; \
		pipdeptree -a > dev/install_log/pipdeptree.txt; \
		pipdeptree -a -r > dev/install_log/pipdeptree_rev.txt; \
		pip-licenses -saud -f markdown -i csv2md --output-file LICENSES_THIRD_PARTY.md; \
		pre-commit install; \
		pre-commit autoupdate;

help:
	@echo "Use \make <target> where <target> is:"
	@echo "  dev           create local dev env"
	@echo "  dev-js        create local dev env using local oreum_core (Jonathan Sedar only)"
	@echo "  lint          run code lint & security checks"
	@echo "  test-dev-env  optional test local dev env numeric packages, v.slow"
	@echo "  uninstall-env remove env (use from parent dir \make -C oreum_copula ...)"


install-env:  ## create mamba (conda) environment
	export PATH=$(MAMBADIR)/bin:$$PATH; \
		if which mamba; then echo "mamba ready"; else make install-mamba; fi
	export PATH=$(MAMBADIR)/bin:$$PATH; \
		export MAMBA_EXE='$(MAMBADIR)/bin/mamba'; \
		export MAMBA_ROOT_PREFIX='$(MAMBADIR)'; \
		mamba update -n base mamba; \
		mamba env create --file condaenv_oreum_copula.yml -y;

install-mamba:  ## get mamba via miniforge, explicitly use bash
	test -f $(MAMBARC) || { echo $(MAMBARCMSG); exit 1; }
	wget $(MAMBADL)/$(MAMBAV) -O $(HOME)/miniforge.sh
	chmod 755 $(HOME)/miniforge.sh
	bash $(HOME)/miniforge.sh -b -p $(MAMBADIR)
	export PATH=$(MAMBADIR)/bin:$$PATH; \
		conda init zsh;
	rm $(HOME)/miniforge.sh

lint: ## run code linters and static security (checks only)
	$(PYTHON) -m pip install bandit interrogate ruff
	ruff check --diff
	ruff format --no-cache --diff
	bandit --config pyproject.toml -r src/
	interrogate --config pyproject.toml src/

slides: ## render slides (and webpdf) and place in publish/ dir
	export PATH=$(MAMBADIR)/envs/oreum_copula/bin:$$PATH; \
		export MAMBA_EXE='$(MAMBADIR)/bin/mamba'; \
		export MAMBA_ROOT_PREFIX='$(MAMBADIR)'; \
		cd notebooks; \
		jupyter nbconvert --config renders/config_slides.py; \
		jupyter nbconvert --config renders/config_webpdf.py
	mv notebooks/renders/000_Intro.slides.html publish/index.html
	mv notebooks/renders/000_Intro.pdf publish/


test-dev-env:  ## test the dev machine install of critial numeric packages
	export PATH=$(MAMBADIR)/bin:$$PATH; \
		export MAMBA_EXE='$(MAMBADIR)/bin/mamba'; \
		export MAMBA_ROOT_PREFIX='$(MAMBADIR)'; \
		$(PYTHON_ENV) -c "import numpy as np; np.test()" > dev/install_log/tests_numpy.txt; \
		$(PYTHON_ENV) -c "import scipy as sp; sp.test()" > dev/install_log/tests_scipy.txt;

tlmgr: ## install tex packages to render notebooks (TeX install is out of scope)
	sudo tlmgr update --all --self
	sudo tlmgr install adjustbox collectbox collection-fontsrecommended \
		enumitem environ eurosym jknapltx mathpazo parskip pdfcol pgf rsfs \
		soul tcolorbox titling trimspaces ucs ulem xcolor

uninstall-env: ## remove mamba env
	export PATH=$(MAMBADIR)/bin:$$PATH; \
		export MAMBA_ROOT_PREFIX='$(MAMBADIR)'; \
		mamba env remove --name oreum_copula -y; \
		mamba clean -ay

uninstall-mamba:  ## nuclear option https://github.com/conda-forge/miniforge?tab=readme-ov-file#uninstall
	mamba shell deinit
	rm -rf $(MAMBADIR)
	rm -rf $(HOME)/.mamba
	rm -f $(HOME)/.mambarc
	rm -f $(HOME)/.mambarc_x86

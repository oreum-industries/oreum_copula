**Oreum Industries**

# Oreum Copula Modelling `oreum_copula`

Examples of Bayesian linear models with copula-based covariance using pymc3.
Primarily for use on client projects.

Notes:

+ Uses a scientific Python stack for scripting and interactive Jupyter Notebooks
+ Hosted privately on
[Oreum Industries' GitHub](https://github.com/oreum-industries/oreum_copula)
+ Developed on a MBP13 2020, 2.3GHz i7, 32GB RAM, MacOS 10.15.7
+ Project began on 2021-08-31
+ The README is MacOS and POSIX oriented
+ See LICENCE.md for licensing and copyright details

# 1. Manual Installation

This project is intended for interactive development and execution.
As at 2021-08-31, the project is manually installed on MacOS: it may be
automated in future.

## 1.1 Pre-req: Continuum Anaconda Python 3.8.* 64bit

Download latest from:
[https://www.continuum.io/downloads](https://www.continuum.io/downloads)

## 1.2 Git clone the repo

Assumes `git` already installed

```zsh
$> git clone https://github.com/oreum-industries/oreum_copula
$> cd oreum_copula/
```

### 1.3 Create virtual environment

Notes:

+ This compound method uses `conda` for main environment and packages,
and `pip` for selected additional packages handled better by pip than conda.
+ Packages might not be the very latest because we want stability for `pymc3`
which is usually in a state of flux, esp w.r.t `theano` and `aesara`.
+ See [cheat sheet of conda commands](https://conda.io/docs/_downloads/conda-cheatsheet.pdf)
+ Use `direnv` on MacOS to automatically run file `.envrc` upon directory open
+ Jupyterlab extensions need [NodeJS](https://nodejs.org/en/) on system

#### 1.3.1 Main environment and packages using conda

```zsh
$> conda update -n base -c defaults conda
$> conda env create --file condaenv_oreum_copula.yml
$> activate oreum_copula
```

#### 1.3.2 Additional packages using pip

Note this includes `oreum_core` which contains several package dependencies.

```zsh
$> ./pip_install.sh
```

#### 1.3.3 Install graphviz to local system, OS dependent

e.g. for MacOS:

```zsh
brew install graphviz
```

#### 1.3.4 Optional jupyterlab extensions

```zsh
$> ./jupyter_install.sh
```

## 1.4 Configs for Local Development

Some notes to help configure local development environment

### 1.4.1 Git config `~/.gitconfig`

```yaml
[user]
    name = <YOUR NAME>
    email = <YOUR EMAIL ADDRESS>
```

### 1.4.2 Jupyter config

Assumes `jupyter` installed, sets defaults

```zsh
$> jupyter notebook --generate-config
$> jupyter qtconsole --generate-config
$> jupyter nbconvert --generate-config
```

### 1.4.3 Local theano config `~/.theanorc`

```yaml
[global]
device=cpu
```

## 1.5 Misc

None

---

# 2. Automatic Installation

Pending creation of `makefile`. This will install onto a Linux server.

---

# 3. Config and Testing

## 3.1 Test installation of scientific packages

Optional tests to confirm good installation of:
BLAS / MKL, numpy, scipy, pymc3, theano

### 3.1.1 BLAS / MKL config

View the BLAS / MKL install

```zsh
$> python -c "import numpy as np; np.__config__.show()"
```

Example output...

```zsh
blas_info:
    libraries = ['cblas', 'blas', 'cblas', 'blas']
    library_dirs = ['/Users/jon/opt/anaconda3/envs/oreum_copula/lib']
    include_dirs = ['/Users/jon/opt/anaconda3/envs/oreum_copula/include']
    language = c
    define_macros = [('HAVE_CBLAS', None)]
blas_opt_info:
    define_macros = [('NO_ATLAS_INFO', 1), ('HAVE_CBLAS', None)]
    libraries = ['cblas', 'blas', 'cblas', 'blas']
    library_dirs = ['/Users/jon/opt/anaconda3/envs/oreum_copula/lib']
    include_dirs = ['/Users/jon/opt/anaconda3/envs/oreum_copula/include']
    language = c
lapack_info:
    libraries = ['lapack', 'blas', 'lapack', 'blas']
    library_dirs = ['/Users/jon/opt/anaconda3/envs/oreum_copula/lib']
    language = f77
lapack_opt_info:
    libraries = ['lapack', 'blas', 'lapack', 'blas', 'cblas', 'blas', 'cblas', 'blas']
    library_dirs = ['/Users/jon/opt/anaconda3/envs/oreum_copula/lib']
    language = c
    define_macros = [('NO_ATLAS_INFO', 1), ('HAVE_CBLAS', None)]
    include_dirs = ['/Users/jon/opt/anaconda3/envs/oreum_copula/include']
Supported SIMD extensions in this NumPy install:
    baseline = SSE,SSE2,SSE3
    found = SSSE3,SSE41,POPCNT,SSE42,AVX,F16C,FMA3,AVX2,AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL
    not found = AVX512_KNL
```

### 3.1.2 numpy

```zsh
$> python -c "import numpy as np; np.test()"
```

Example output...

```zsh
-- Docs: https://docs.pytest.org/en/latest/warnings.html
9902 passed, 435 skipped, 180 deselected, 17 xfailed, 3 xpassed, 2 warnings in 82.04s (0:01:22)
```

### 3.1.3 scipy

```zsh
$> python -c "import scipy as sp; sp.test()"
```

Example output...

```zsh
25474 passed, 2003 skipped, 10989 deselected, 76 xfailed, 5 xpassed, 11 warnings in 338.80s (0:05:38)
```

### 3.1.4 pymc3

Method 1:

```zsh
$> python -m pytest -xv --cov=pymc3 --cov-report=html pymc3/
```

Method 2:

Takes a while on my MBP13 2020

```zsh
$> python -c "import pymc3 as pm; pm.test()"
```

### 3.1.5 theano

Takes forever, see
[installation docs](http://deeplearning.net/software/theano/install_macos.html)

Quicker:

```zsh
$> python -c "import theano; theano.test()"
```

Alternative (takes ~3 hours)

```zsh
$> theano-nose -s
```

---

# 4. Data

See `data/README_DATA.md`

---

# 5. General Notes

None

---
**Oreum OÜ &copy; 2021**
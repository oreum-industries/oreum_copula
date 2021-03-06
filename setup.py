# setup.py
# copyright 2021 Oreum OÜ
import re
from codecs import open
from os.path import dirname, join, realpath
from setuptools import find_packages, setup
from setuptools.command.develop import develop
from setuptools.command.install import install
from subprocess import check_call

DISTNAME = "oreum_copula"
DESCRIPTION = "Copula demos for use on projects by Oreum Industries"
AUTHOR = "Jonathan Sedar"
AUTHOR_EMAIL = "jonathan.sedar@oreum.io"
URL = "https://github.com/oreum-industries/oreum_copula"
LICENSE = "Proprietary"

CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Financial and Insurance Industry",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Topic :: Scientific/Engineering :: Visualization",
    "Operating System :: MacOS",
    "Operating System :: Unix",
    "License :: Other/Proprietary License"
]

PROJECT_ROOT = dirname(realpath(__file__))

# Get the long description from the README file
with open(join(PROJECT_ROOT, "README.md"), encoding="utf-8") as buff:
    LONG_DESCRIPTION = buff.read()

REQUIREMENTS_FILE = join(PROJECT_ROOT, "requirements.txt")

with open(REQUIREMENTS_FILE) as f:
    install_reqs = f.read().splitlines()

def get_version():
    VERSIONFILE = join(DISTNAME, "__init__.py")
    lines = open(VERSIONFILE).readlines()
    version_regex = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in lines:
        mo = re.search(version_regex, line, re.M)
        if mo:
            return mo.group(1)
    raise RuntimeError(f"Unable to find version in {VERSIONFILE}.")

if __name__ == "__main__":
    setup(
        name=DISTNAME,
        version=get_version(),
        maintainer=AUTHOR,
        maintainer_email=AUTHOR_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        url=URL,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/x-md",
        packages=find_packages(),
        include_package_data=True,
        classifiers=CLASSIFIERS,
        python_requires=">=3.9",
        install_requires=install_reqs
    )
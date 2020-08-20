from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import codecs
import os
import sys
import re

import figure_generation

here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    # intentionally *not* adding an encoding option to open
    return codecs.open(os.path.join(here, *parts), 'r').read()

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

long_description = read('README.md')

class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errcode = pytest.main(self.test_args)
        sys.exit(errcode)

setup(
    name='deepcell-kiosk-figure-generation',
    version=find_version('figure_generation','__init__.py'),
    url='https://deepcell.org',
    license='Apache Software License (Modified)',
    author='Dylan Bannon',
    tests_require=['pytest'],
    install_requires=['matplotlib>=3.1.1,<4.0.0',
                      'pandas>=0.25.0,<0.26.0',
                     ],
    cmdclass={'test': PyTest},
    author_email='bbannon@caltech.edu',
    description='Figure generation from Deepcell Kiosk benchmarking data',
    long_description=long_description,
    packages=['figure_generation'],
    include_package_data=True,
    platforms='any',
    test_suite='test.test_figure_generation',
    classifiers = [
        'Programming Language :: Python',
        'Development Status :: 5 - Production/Stable',
        'Natural Language :: English',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License (Modified)',
        'Operating System :: OS Independent',
        'Topic :: Scientific Engineering :: Medical Science Apps.',
        'Topic :: Scientific Engineering :: Visualization',
        ],
    extras_require={
        'testing': [
            'pytest>=5.1.3,<6.0.0',
            'pylint>=2.3.1,<3.0.0',
            ],
        'docs': [
            'Sphinx>=2.2.0,<3.0.0',
            'sphinx_rtd_theme>=0.4.3,<0.5.0',
            ],
    }
)

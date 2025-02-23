from setuptools import setup, find_packages

import codecs
import os.path

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='ozone',
    version=get_version('ozone/__init__.py'),
    author='Author name',
    author_email='msperry@ucsd.edu.com',
    license='LGPLv3+',
    keywords='python project template repository package',
    url='http://github.com/LSDOlab/ozone',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    python_requires='>=3.9',
    platforms=['any'],
    install_requires=[
        'csdl_alpha @ git+https://github.com/LSDOlab/CSDL_alpha.git',
        'modopt @ git+https://github.com/LSDOlab/modopt.git',
        'numpy',
        'scipy',
        # 'pyslsqp',
        'pytest',
        'smt',
        'seaborn',
        'myst-nb',
        'sphinx==5.3.0',
        'sphinx_rtd_theme',
        'sphinx-copybutton',
        'sphinx-autoapi==2.1.0',
        'astroid==2.15.5',
        'numpydoc',
        'gitpython',
        'sphinxcontrib-collections @ git+https://github.com/anugrahjo/sphinx-collections.git', # 'sphinx-collections',
        'sphinxcontrib-bibtex',
        'setuptools',
        'wheel',
        'twine',
    ],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Topic :: Documentation',
        'Topic :: Documentation :: Sphinx',
        'Topic :: Software Development',
        'Topic :: Software Development :: Documentation',
        'Topic :: Software Development :: Testing',
        'Topic :: Software Development :: Libraries',
    ],
)

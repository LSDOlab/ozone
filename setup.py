from distutils.core import setup

setup(
    name='ozone',
    version='0.0.1',
    packages=[
        'ozone',
    ],
    install_requires=[
        'numpy',
        'scipy',
        'modopt @ git+https://github.com/LSDOlab/modopt.git',
        'csdl @ git+https://github.com/LSDOlab/csdl.git',
        'python_csdl_backend @ git+https://github.com/LSDOlab/python_csdl_backend.git',
        'pytest',
        'matplotlib',
    ],
)

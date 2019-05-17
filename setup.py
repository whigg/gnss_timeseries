from setuptools import setup
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='gnss_timeseries',
    version='19.02',  # YYYYY.MM = year and month; b1 = beta-1 release
    description='Time series of a set of variables and its statistics. Ring '
                'buffer of a group variables represented as numpy arrays.',
    long_description=long_description,  # [1]
    # The project's main homepage.
    # url='https://github.com/pypa/sampleproject',
    # Author details
    author='Francisco del Campo R.',
    author_email='fdelcampo@csn.uchile.cl',
    license='MIT',  # [2]
    python_requires='>=3.5, !=3.7.*',
    install_requires=['numpy',
                      'matplotlib',
                      'scipy',
                      'geoproj'],
    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    # packages=find_packages(exclude=['contrib', 'docs', 'examples']),
    packages=['gnss_timeseries'],
    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in gnss_plots.in as well.
    package_dir={'gnss_timeseries': 'gnss_timeseries'},
    package_data={'gnss_timeseries': ['./t_student_table',]},
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',  # [3]
        # 'Development Status :: 5 - Production/Stable',  # [3]
        # Indicate who your project is intended for
        'Intended Audience :: Science/Research :: Developers',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering',
        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 3 or both.
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
        ],
    # What does your project relate to?
    keywords='GNSS Geodesy time-series',
    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    # entry_points={
    #     'console_scripts': [
    #         'example=examples:main',
    #     ],
    # },
    )

# [1] on the PyPI the field "long_description" will be used
#     as the description of the package on the website.
# [2] Specifying a licence is important for Open Source software
# [3] Maturity of package.

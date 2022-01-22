from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required_list = f.read().splitlines()

setup(
    name="some_tools",
    version="1.0.6",
    author="Matteo Bagagli",
    author_email="matteo.bagagli@erdw.ethz.ch",
    description="tools and routines for the SOME project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mbagagli/some_tools",
    python_requires='>=3.7',
    install_requires=required_list,
    setup_requires=['wheel'],
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Intended Audience :: Science/Research",
    ],
    # entry_points={
    #     'bin': [
    #         'sometools_traceplot.py=filterpicker.cli.obspy_script:main',
    #     ],
    # },
    include_package_data=True,
    zip_safe=False,
)

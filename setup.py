from setuptools import setup, find_packages  # Extension

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required_list = f.read().splitlines()


# ==== If C extension
# cmodule = Extension('host/src/host_clib',
#                     sources=['host/src/host_clib.c'],
#                     extra_compile_args=["-O3"])

setup(
    name="some_tools",
    version="0.0.1",
    author="Matteo Bagagli",
    author_email="matteo.bagagli@ingv",
    description="tools and routines for the SOME project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mbagagli/some_tools",
    python_requires='>=3.7',
    install_requires=required_list,
    setup_requires=['wheel'],
    packages=find_packages(),
    # package_data={"some_tools": ['src/*.c']},     # if C implementations
    # ext_modules=[cmodule],                        # if C implementations
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

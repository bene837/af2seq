from setuptools import setup, find_packages


setup_info = dict(
    name='af2seq',
    version='0.0.1',
    author='Benedict Wolf, Casper Goverde',

    description='de novo protein design using AlphaFold',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],

    # Package info
    packages=find_packages(),
    #install_requires=['python_version == "3.8"',
    #]
)

setup(**setup_info)
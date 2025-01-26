from setuptools import setup, find_packages

descr = "A python port of the Hirem R package by Jonas Crevecoeur et al"
name = 'hirem'
url = "https://github.com/carpenn/hirem"
version='0.1.2' 

setup(
    name=name,
    version=version,
    url=url,
    packages=find_packages(),
    install_requires=[],
    author="Nigel Carpenter",
    author_email="nigel.carpenter1@googlemail.com",
    description=descr,
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    include_package_data=True,
    package_data={'':['data/*.csv']},
)
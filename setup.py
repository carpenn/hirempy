from setuptools import setup, find_packages

setup(
    name="hirempy",
    version="0.1",
    packages=find_packages(),
    install_requires=[],
    author="Nigel Carpenter",
    author_email="nigel.carpenter1@googlemail.com",
    description="A python port of the Hirem R package by Jonas Crevecoeur et al",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/carpenn/hirempy",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
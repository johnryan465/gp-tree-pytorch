import setuptools

setuptools.setup(
    name="gptree",
    version="0.0.1",
    author="John Ryan",
    project_urls={
        "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "gptree"},
    packages=setuptools.find_packages(where="gptree"),
    python_requires=">=3.7",
)
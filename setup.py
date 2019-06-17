import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
	name="FTools",
    version="0.0.1",
    author="Zhilong Fang",
    author_email="fangzl@mit.edu",
    description="Tool box for Fang Zhilong personal usage",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zfang-slim/FTools.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)


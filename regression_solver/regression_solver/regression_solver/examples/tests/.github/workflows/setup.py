from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="regression-solver",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Comprehensive regression solver with linearization support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/regression-solver",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0", "pytest-cov"],
    },
)

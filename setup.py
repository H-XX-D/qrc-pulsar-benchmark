from setuptools import setup, find_packages

setup(
    name="qrc-pulsar-bench",
    version="1.0.0",
    author="Todd Hendrixx (H-XX-D)",
    author_email="todd@hendrixxdesign.com",
    description="High-precision relational pulsar timing & multi-body benchmark suite grounded in Quantum Relational Causality (QRC)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/H-XX-D/qrc-pulsar-benchmark",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
    ],
    entry_points={
        "console_scripts": [
            "qrc-bench=qrc_pulsar_bench.cli:main",
        ],
    },
)

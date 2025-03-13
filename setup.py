from setuptools import setup, find_packages

setup(
    name="afq_insight_experiments",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "seaborn>=0.11.0",
        "afqinsight>=0.3.0",
        "jupyter>=1.0.0",
        "ipywidgets>=7.6.0",
        "tqdm>=4.60.0",
    ],
    author="Sam Chou",
    author_email="sam@thechous.com",
    description="Autoencoder experiments for AFQ-Insight, For UW Neuroinformatics R&D Group",
    keywords="autoencoder, afq, neuroimaging, diffusion MRI",
    url="https://github.com/yourusername/AFQ-Insight-Autoencoder-Experiments",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
) 
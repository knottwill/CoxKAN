import setuptools

# Load the long_description from README.md
with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

requirements = [
    'pykan==0.0.2',
    'lifelines>=0.28.0',
    'torch>=2.3.1',
    'tqdm',
    'optuna>=3.6.1',
    'torchtuples==0.2.2',
    'scikit-learn>=1.5.0',
    'feather-format>=0.4.0',
    'h5py>=2.9.0',
    'numba>=0.44',
    'requests>=2.22.0',
    'py7zr>=0.11.3',
    'scipy>=1.13.1',
]

setuptools.setup(
    name="coxkan",
    version="0.0.2",
    author="William Knottenbelt",
    author_email="knottenbeltwill@gmail.com",
    url="https://github.com/knottwill/CoxKAN",
    description="CoxKAN: Kolmogorov-Arnold Networks for Survival Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(include=['coxkan', 'coxkan.*']),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    license='Apache License 2.0',
    python_requires='>=3.6',
)
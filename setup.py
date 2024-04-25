import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="esinet",
    version="0.3.0",
    author="Lukas Hecker",
    author_email="lukas_hecker@web.de",
    description="Solves the M/EEG inverse problem using artificial neural networks with Python 3 and the MNE library.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LukeTheHecker/esinet",
    packages=setuptools.find_packages(),
    install_requires=['tensorflow','mne', 'scipy', 'colorednoise', 'joblib', 'matplotlib', 'pyvista', 'pyvistaqt', 'vtk', 'tqdm', 'pytest', 'dill', 'scikit-learn', 'pandas'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

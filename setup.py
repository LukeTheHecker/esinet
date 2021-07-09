import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="esinet",
    version="0.0.13",
    author="Lukas Hecker",
    author_email="lukas_hecker@web.de",
    description="Solve the M/EEG inverse problem using artificial neural networks with Python 3 and the MNE library.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LukeTheHecker/esinet",
    packages=setuptools.find_packages(),
    install_requires=['tensorflow>=2.4.1','mne>=0.22.0', 'scipy', 'colorednoise', 'joblib', 'jupyter', 'ipykernel', 'matplotlib', 'pyvista>=0.27.4', 'pyvistaqt>=0.3.0', 'vtk>=9.0.1', 'tqdm'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8.3',
)

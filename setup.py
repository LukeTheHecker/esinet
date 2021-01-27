import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ESINet", # Replace with your own username
    version="0.0.8",
    author="Lukas Hecker",
    author_email="lukas_hecker@web.de",
    description="Solve the M/EEG inverse problem using artificial neural networks with Python 3 and the MNE library.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LukeTheHecker/ESINet",
    packages=setuptools.find_packages(),
    install_requires=['tensorflow>=2.4.1','mne>=0.22.0', 'scipy', 'colorednoise', 'joblib', 'jupyter', 'ipykernel', 'matplotlib', 'pyvista>=0.24', 'pyvistaqt>=0.2.0', 'vtk>=9.0.1', 'tqdm'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8.3',
)

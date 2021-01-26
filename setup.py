import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ESINet", # Replace with your own username
    version="0.0.6",
    author="Lukas Hecker",
    author_email="lukas_hecker@web.de",
    description="Solve the M/EEG inverse problem using artificial neural networks with Python 3 and the MNE library.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LukeTheHecker/ESINet",
    packages=setuptools.find_packages(),
    install_requires=['tensorflow','mne', 'scipy', 'colorednoise', 'joblib'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8.3',
)

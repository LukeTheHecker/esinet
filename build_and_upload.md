# Built, test and upload routine

## Update your package information
1. Add changes to the changelog and 
2. update the version number in setup.py so people can track the development of the package.

## activate your dev environment
```
workon esienv
```

## Build the package

```
# Update/ install build
py -m pip install --upgrade build
py -m build --wheel
```

## Check whether your builds are fine

```
twine check dist/*
```

## Test your package locally

Run the batch file to test your local package:

```
test_build.bat
```

## Upload the package to pypi

```
twine upload dist/*
```

## Test the remote package

```
test_build2.bat
```
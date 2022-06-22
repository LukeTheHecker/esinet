# Built, test and upload routine

## Update your package information
1. Add changes to the changelog and 
2. update the version number in *setup.py* so people can track the development of the package.

## Activate your dev environment
```
workon esienv
```

## Test locally
First, test your code to see if everything is working fine.

```
test_local.bat
```

## Commit changes
Commit your changes.

```
git commit -m "commit message"
```

## Build the package and check them

```
py -m build --wheel
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
# Some reminders on how to handle the package with PyPi

## Make changes
Make some changes to your code

## Prepare

* Increment package version you can do that in the setup.py file

* Open your console and activate a suitable environment:

```
workon esienv
```

* add changes to the changelog 

## Test it


## Pack it
Pack your package using:
```
python setup.py sdist bdist_wheel
```

## Upload the Package
```
python -m twine upload  dist/*
```
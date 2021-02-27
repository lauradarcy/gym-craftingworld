## things to update for new version:
* **please ensure all print statements are removed from the package before uploading**
* update version number in setup.py
* update version number in docs/requirements.txt

## workflow to update github and pypi package:
* commit to github
* check pass travis build
* update readthedocs AFTER uploading new pypi version

## steps to upload new pypi version
* activate conda env with wheel
* run `python3 setup.py sdist bdist_wheel`
* run `python3 -m twine upload dist/*`
* finally update readthedocs

## To create the Conda Package:

* `conda env export --no-builds -n DimensionReduction -f environment.yml`
  * Remove the "prefix: ..." line from environment.yml
  * Note: This captures some platform-specific dependencies.  In the end,
    I went with a hand-rolled list of dependencies.  Also I removed almost all
    version numbers as those vary platform to platform in some cases.
* Cleanup directory (.git, etc.) 
   * `rm -rf .git; rm -rf .ipynb_checkpoints; rm -rf __pycache__; rm -rf .DS_Store; rm .gitignore`
* Create a zip file of the directory `zip -r dimension-reduction.zip dimension-reduction`
* `anaconda upload --force --package dimension-reduction --package-type file --version 1 --summary "Course materials" dimension-reduction.zip`
* If testing on the same machine as creating: `source deactivate; conda env remove -n DimensionReduction -y`


## Install (unix-like):

* Change to a "safe" directory
* `anaconda download SomewhatUseful/dimension-reduction`
   * On OS X may get a UTF-8 error, [it can be ignored](https://conda.io/docs/user-guide/troubleshooting.html)
* `unzip dimension-reduction.zip`
* `cd dimension-reduction`
* `conda env create -f environment.yml` to make a new env OR `conda env update -f environment.yml` to
  install into the current environment.
* `conda activate DimensionReduction`
* `jupyter notebook`

## Install (windows):

* Open `anaconda prompt` from the menu
* `anaconda download SomewhatUseful/dimension-reduction`
* Extract all contents of the file
* `cd dimension-reduction`
* `cd dimension-reduction` (yes, twice...)
* `conda env create -f environment.yml`
* `pip install fastdtw`
* `conda activate DimensionReduction`
* `jupyter notebook`

## Some resources:
* https://docs.anaconda.com/anaconda-cloud/user-guide/tasks/work-with-other-files
* 

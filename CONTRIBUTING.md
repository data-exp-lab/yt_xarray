
# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

## Types of Contributions

### Report Bugs


Report bugs at https://github.com/chrishavlin/yt_xarray/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

### Write Documentation

yt_xarray could always use more documentation, whether as part of the
official yt_xarray docs, in docstrings, or even on the web in blog posts,
articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at https://github.com/chrishavlin/yt_xarray/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

## Get Started!

Ready to contribute? Here's how to set up `yt_xarray` for local development.

1. Fork the `yt_xarray` repo on GitHub.
2. Clone your fork locally:
```
$ git clone git@github.com:your_name_here/yt_xarray.git
```

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed, this is how you set up your fork for local development::
```
$ mkvirtualenv yt_xarray
$ cd yt_xarray/
$ python -m pip install -e .
$ python -m pip install -r requirements_dev.txt
```
4. (optional) Setup pre-commit
```
$ pre-commit install
```

5. Create a branch for local development::

```
$ git checkout -b name-of-your-bugfix-or-feature
```
Now you can make your changes locally.

6. When you're done making changes, check that your changes pass the
   tests:

```
$ pytest
```
To test multiple python versions, you can use `tox`:

```
$ tox
```


7. Commit your changes and push your branch to GitHub:

```
$ git add .
$ git commit -m "Your detailed description of your changes."
```
If you've installed pre-commit, then pre-commit will run your changes through
some style checks. It will try to fix files if needed. If it finds errors, you
will need to re-add those files (after fixing it if pre-commit could not do so
automatically) and then commit again.

Now your branch is ready to push:

```
$ git push origin name-of-your-bugfix-or-feature
```

8. Submit a pull request through the GitHub website.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring.
3. The pull request should pass all automated tests.

## Tips

To run a subset of tests:

$ pytest tests.test_yt_xarray


## Deploying

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed (including an entry in HISTORY.md).

```
$ bump2version patch # possible: major / minor / patch
$ git push
$ git push --tags
```
github will then push to PyPI if tests pass.

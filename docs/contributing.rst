Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
######################

Report Bugs
***********

Report bugs at https://github.com/data-exp-lab/yt_xarray/issues

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
********

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
******************

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
*******************

yt_xarray could always use more documentation, whether as part of the
official yt_xarray docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
***************
The best way to send feedback is to file an issue at https://github.com/data-exp-lab/yt_xarray/issues

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
############

Ready to contribute? Here's how to set up `yt_xarray` for local development.

1. Fork the `yt_xarray` repo on GitHub.
2. Clone your fork locally::


    $ git clone git@github.com:your_name_here/yt_xarray.git


3. Install your local copy into a virtual environment. Using a plain `venv`
   (any other environment manager works just as well)::

    $ cd yt_xarray/
    $ python -m venv .venv
    $ source .venv/bin/activate
    $ python -m pip install -e ".[full,test]"

   The `full` extra pulls in the optional runtime dependencies (netCDF4, scipy,
   dask, cf_xarray) and `test` adds pytest and friends. To also run the type
   checker locally, use ``".[typecheck]"`` instead, which includes both.

4. (optional) Setup pre-commit. The hooks run `ruff check --fix` and
   `ruff format`, the same checks that pre-commit.ci runs on pull requests::

    $ python -m pip install pre-commit
    $ pre-commit install

5. Create a branch for local development::


    $ git checkout -b name-of-your-bugfix-or-feature

Now you can make your changes locally.

6. When you're done making changes, check that your changes pass the
   tests::


    $ pytest yt_xarray

7. (optional) Run the type checker. This requires the `typecheck` extra from
   step 3 and is also run by CI on pull requests::


    $ mypy yt_xarray

8. Commit your changes and push your branch to GitHub::


    $ git add .
    $ git commit -m "Your detailed description of your changes."

If you've installed pre-commit, then pre-commit will run your changes through
some style checks. It will try to fix files if needed. If it finds errors, you
will need to re-add those files (after fixing it if pre-commit could not do so
automatically) and then commit again.

Now your branch is ready to push::

    $ git push origin name-of-your-bugfix-or-feature

9. Submit a pull request through the GitHub website.

Pull Request Guidelines
#######################

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring.
3. The pull request should pass all automated tests.

Tips
####

To run a subset of tests::

    $ pytest yt_xarray/tests/test_accesor.py
    $ pytest yt_xarray/tests/test_accesor.py::test_name


Deploying
#########

A reminder for the maintainers on how to deploy.

First, make sure that:

* all your changes are committed.
* the version in `yt_xarray/__init__.py` matches the version you are releasing.
* HISTORY.md is finalized and merged into `main` **before** you tag: the tagged
  commit must contain a ``## <version>`` heading (without the leading ``v``,
  e.g. ``## 1.1.2``) with the release notes for this version. The release
  workflow reads HISTORY.md from the tagged commit and fails before building
  any wheels if that section is missing or empty, so a leftover
  ``## 1.1.2dev`` heading will stop the release.

Then, double check your `main` branch matches upstream (assuming that `git remote -v` lists
`git@github.com:data-exp-lab/yt_xarray.git` as the upstream repo)::

    $ git fetch --all
    $ git checkout main
    $ git rebase upstream/main

Confirm that HISTORY.md at this commit has the finalized ``## 1.1.2`` section
(if not, fix it, merge it, and rebase again before continuing). Then create a
tag for your new release. For example, if you are releasing version `1.1.2`::


    $ git tag v1.1.2

Now push that tag to the upstream repo::

    $ git push upstream v1.1.2

This triggers a single Github action (`.github/workflows/publish-to-pypi.yml`)
that runs the following jobs in order:

1. extract the release notes for this version from HISTORY.md (fails if missing).
2. build the sdist and wheel and run `twine check`.
3. publish the distributions to PyPI via trusted publishing.
4. create a draft release on Github with the HISTORY.md section as the body and
   the sdist and wheel attached as assets.

The draft is only created if the PyPI upload succeeds. Once the action finishes,
go to https://github.com/data-exp-lab/yt_xarray/releases, review the draft, and
hit publish.

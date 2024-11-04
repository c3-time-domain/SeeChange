## Contribution guide

Here are some guidelines for contributing to this project.


### Style recommendations

This project does not enforce rigid style rules, 
and does not apply pre-commit hooks. 
However, we recommend following the PEP8 style guide. 
Some exceptions are made to match our style preferences, 
e.g., the character line length limit is set to 120,  
and spacing is sometimes added inside parenthesis 
in function arguments and calculations are added 
to improve readability where needed.  


### Pull requests

To contribute to the code, please first fork the repository,
and when ready to start working on a new feature, 
make a new branch and set it to track the same branch name
on your own remote repository. 
This should immediately allow you to open a pull request
on the main repo at [https://github.com/c3-time-domain/SeeChange]. 

Pull requests should have a short description of the changes,
why they are required and some guidance to the reviewer on 
what to look out for. 

As a small project, we institute a policy of one reviewer per pull request. 
One of the core developers will be happy to review the code 
before merging it into the main branch.
Also note that a suite of automated tests should pass
before merging any new code. 


### Testing

Tests run in a few folders under `tests`, using the `pytest` package. 
These can be run on a local machine,
by calling `pytest` from the main directory.
Tests can also be run in a containerized environment, 
which is much closer to the production environment (or the github actions environment).
See the `docs/setup.md` file for more details about running environments. 

Every time a pull request is opened (and for each commit pushed to it), 
we run a set of automated tests, using a separate github action 
for each folder under `tests`. 
Currently, this includes `improc`, `models`, `pipeline`, and `util`. 
Tests running on github use a docker container very similar 
to the production environment, and thus should be a good indication
of whether the code will run in production.
If tests appear to run or not run on a local install vs. a github action,
this could be due to one of the following reasons:
 - Version mismatch of Python or any installed packages or standalone executables (e.g., sextractor). 
 - Missing environment variables (see `docs/setup.md` for more details).
 - Missing or different config files (see `docs/setup.md` for more details).
 - Migration has not been performed on the local database, but the database schema has changed. 
 - Remaining database objects or files on disk have been left over from previous test runs. 
   While we try to clean up after all tests, sometimes residul objects remain. 
   The github actions tests always run on a new container, so this is not an issue there.

Please make sure to write tests for new features. 
It should not be necessary to say this, 
but in our experience, code that is not tested is often broken.


### Documentation

Please make sure to add "numpy style" docstrings to all functions. 
Ommitting the Parameters or Returns clause for simple functions 
is acceptable as long as the variables are self-explanatory.


### Coding assistance

We (...well, the original author of this paragraph; the current main
developer acutally disagrees with this enitre section...) encourage
developers to use a modern IDE such as PyCharm or VSCode.  This helps
mitigate spelling mistakes, and highlights style guide violations that
are otherwise not enforced by the code base.

Using AI methods for code completion is encouraged, 
in fact, some of the developers of this code base 
have given up on typing code altogether, 
and are using copilot to read their minds and 
write the code for them.

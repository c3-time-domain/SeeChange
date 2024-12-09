## Contribution guide

Here are some guidelines for contributing to this project.

### Licensing

The pipeline is licensed under the LBNL variant of the BSD 3-clause open source license— see the file `LICENSE` at the top level of the project.

### Style

We run the `ruff` linter to check for errors and to enforce some style before running automated tests on pull requests.  The file `ruff.toml` lists exactly which ruff rules are enforced.  Many of the rules are checking for errors (local variables used before they're defined, imports that are never used, local variables assigned but never used, deprecated numpy routines, etc.).  However, it also does enforce a handful of style rules, including (but not limited to):

- 120-character width lines.  (If you really must have a longer line, e.g for storing a gigantic base64 encoded string literal, add `  # noqa: E501` to the end of the line.)
- 4-character indentation steps
- indention with spaces, not tabs (tabs are evil)
- 2 blank lines before top-level functions or classes
- no whitespace at the end of a line
- must have a space after the commas that separate function arguments
- semicolons at the end of lines because you are going back and forth between JavaScript and python and forget which language you're currently in.  (Mixing up `for...in` and `for...of` in JavaScript is a more serious consequence of this coding mode.)
- Not using an f-string where there are no actual substitutions.
- Docstrings must start with a single summary line in the same line as the opening """ of the docstring.


### Issues

If you intend to submit a pull request to the main SeeChange archive, please post requests for fixes or features, including things you're currently working on, to the Issues in the github archive (https://github.com/c3-time-domain/SeeChange/).  There are a *lot* of open issues right now, because the code is under active development.  Check to see if there's already an issue for what you want to work on.


### Pull requests

To contribute to the code, please first fork the repository, and when ready to start working on a new feature, make a new branch and set it to track the same branch name on your own remote repository.  This should immediately allow you to open a pull request on the main repo at [https://github.com/c3-time-domain/SeeChange].

Pull requests should have a short description of the changes, why they are required and some guidance to the reviewer on what to look out for.

As a small project, we institute a policy of one reviewer per pull request.  One of the core developers will be happy to review the code before merging it into the main branch.  Also note that a suite of automated tests should pass before merging any new code.


### Testing

Make sure to write tests for new features, and to update tests if you've added or changed to the functionality of existing functions or methods.  All the tests are in the `tests` subdirectory.

For more information about running the tests, see Testing.

Please make sure to write tests for new features.  It should not be necessary to say this, but in our experience, code that is not tested is often broken.  (Code that is tested is less often broken....)


### Documentation

Please make sure to add "numpy style" docstrings to all functions.  Ommitting the Parameters or Returns clause for simple functions is acceptable as long as the variables are self-explanatory.


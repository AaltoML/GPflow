# How to make a new GPflow release

1. Check that [RELEASE.md](RELEASE.md) contains the up-to-date release notes for the next release.
   - They should cover all (non-GitHub-related) commits (PRs) on the `develop` branch since the most recent release.
   - They should make clear to users whether they might benefit from this release and what backwards incompatibilities they might face.

2. Bump the version numbers in the `develop` branch, in the VERSION file **and** in doc/source/conf.py ([example PR: #1395](https://github.com/GPflow/GPflow/pull/1395)).
   Copy the RELEASE.md template for the following release-in-progress.

3. Create a release PR from `develop` to `master`.
   - **Make a merge commit. DO NOT SQUASH-MERGE.**
   - If you squash-merge, `master` will be *ahead* of develop (by the squash-merge commit). This means we’ll end up with merge conflicts at the following release!
   - [example PR: #1396](https://github.com/GPflow/GPflow/pull/1396)

4. Go to the [release page on GitHub](https://github.com/GPflow/GPflow/releases/new) and create a release for a tag “v{VERSION}” (e.g., for version 2.1.3 the tag needs to be `v2.1.3`) to `master` branch. Copy the release notes into the description field!
   - [example release: v2.0.0](https://github.com/GPflow/GPflow/releases/tag/v2.0.0)

5. You are almost done now! Go to https://circleci.com and monitor that tests for your newly-created tag passed and the job for pushing the pip package succeeded. CircleCI matches on the “v{VERSION}” tag to kick-start the release process.
   - [example CI workflow: 2434](https://app.circleci.com/pipelines/github/GPflow/GPflow/2434/workflows/f1274aa7-18c6-45a3-8d59-cab573305b64)

6. Take a break: before you continue, wait until the new release [shows up on PyPi](https://pypi.org/project/gpflow/#history).

Context: Circleci of our doc-repo “GPflow/docs.git” is automatically triggered to re-build the ReadTheDocs documentation when pushed to gpflow/develop branch. This is not the case when pushing to gpflow/master. 
Therefore, after pushing the release to PyPi, we need to trigger the circleci build for docs/master. To do this, go to doc-repo “GPflow/docs.git” master branch and update in .circleci/config.yml (line 14) the VERSION number. Commit and push this change to the master branch, and the docs will build. 

```bash
git clone -b master git@github.com:GPflow/docs.git
cd docs
open .circleci/config.yml
```

Update the version number of `pip install -q gpflow=={x.x.x}` in line 14. Commit change and push to master.

Done done! Go and celebrate our hard work :)


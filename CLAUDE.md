# git-brew

## Releases

The version lives in `Cargo.toml`; `Cargo.lock` and the conda recipe
carry copies, and CI refuses a release tag that disagrees with either
manifest. `cargo release` keeps all three in step, so none of them is
edited by hand.

After a change that users would notice lands on `main` (a fix, a new
flag, a change in behaviour), suggest a release and the bump level.
While on 0.x, a breaking change to the command line, the `brew.prefix`
config key or the refusal rules takes a minor bump; everything else
takes a patch bump. CI, documentation and refactors do not prompt a
release on their own.

A change to the packaging alone (a recipe requirement, a run
constraint) raises `build: number` in `recipe/recipe.yaml` instead of
the version, so `pixi global update` picks it up without a new release
number.

To release, run `cargo release <level>` first and show the dry run.
`--execute` then commits and tags locally; ask before pushing either,
as the tag push alone publishes the packages and the GitHub release.

# ☕ git-brew

[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/license/mit)
[![CUDA](https://img.shields.io/badge/CUDA-sm__75%2B-76B900.svg)](#requirements)
[![Rust](https://img.shields.io/badge/Rust-stable-dea584.svg)](rust-toolchain.toml)

Every commit deserves to start the day with a little `c0ffee`!

Brute-force a chosen hex prefix onto a git commit hash, on the GPU.
Point it at `HEAD`, name a prefix, and it rewrites the commit so its
SHA-1 starts with your digits - by appending invisible whitespace to
the message, so nothing you can see changes.

<!-- TODO: record on the GB10 with `vhs docs/demo.tape`, then
     uncomment:
![git brew finding a c0ffee prefix](docs/demo.gif)
-->

```
$ git brew c0ffee
searching for prefix 'c0ffee': ~2^24 of 2^48 candidates expected

9d000786cdc6fcfb772ec03bde1bb51e5fed37a3 -> c0ffee4ffc212a7760b30c407c87ad2ce16807c8
```

The commit message, your tree, your author and date all stay exactly
as they were. The only change is a run of trailing spaces and tabs on
the last line, which every git UI ignores.

## Why GPU accelerated?
Smaller prefixes are easy to find on a CPU, but the search space grows
exponentially with each hex digit. A 12-digit prefix is a 2^48 search,
which is a lot of work for a CPU. A modern GPU can do it in hours.

While this isn't the usual use case of a vanity commit hasher, being that
the shortest prefix that GitHub shows is 7 hex digits, it is a fun way to
show off a GPU's SHA-1 throughput and put that extra compute to use!

## Requirements

- An NVIDIA GPU (sm_75 / Turing or newer) and its driver.
- To build: the CUDA toolkit (developed against CUDA 13.0), a Rust
  toolchain (`rustup`, stable), CMake >= 3.24, Ninja, and a C++17
  compiler. A built binary needs none of these - the CUDA runtime is
  linked in statically, so only the driver matters at run time.

## Build and install

```
cargo build --release
cp target/release/git-brew ~/.local/bin/   # anywhere on PATH
```

Being on `PATH` is all it takes: git's subcommand dispatch then gives
you `git brew <prefix>` for free, no config anywhere.

The default build is a fatbinary: SASS for every architecture CUDA 13
can target (sm_75 through sm_120, with minor-forward compatibility
filling the gaps - a GB10's sm_121 runs the sm_120 cubin) plus sm_120
PTX for future majors. One binary runs on any supported GPU. To narrow
the build to a single architecture for faster dev iteration:

```
GIT_BREW_CUDA_ARCH=121 cargo build --release   # e.g. GB10 only
```

## Usage

```
git brew [--dry-run] [--force] [<prefix>]
```

- `<prefix>` is 1 to 12 hex digits, case-insensitive. Omit it and the
  tool reads `git config brew.prefix`, which makes a per-repo default
  a one-liner:

  ```
  git config brew.prefix c0ffee
  ```

- `--dry-run` searches and verifies, prints the hash it would produce,
  and writes nothing.
- `--force` amends even when `HEAD` is already on a remote-tracking
  ref. Reach for it deliberately.

An alias for the common "commit, then brew it" path:

```
git config --global alias.cb '!f() { git commit "$@" && git brew; }; f'
```

The shell-function wrapper is load-bearing: a bare
`!git commit "$@" && ...` alias appends the arguments twice, because
git passes them through *and* `$@` expands.

## A worked example

Commit as normal, then point the tool at the result:

```
$ git commit -m "Add the frobnicator"
$ git log --oneline -1
9d00078 Add the frobnicator

$ git brew c0ffee
searching for prefix 'c0ffee': ~2^24 of 2^48 candidates expected

9d000786cdc6fcfb772ec03bde1bb51e5fed37a3 -> c0ffee4ffc212a7760b30c407c87ad2ce16807c8

$ git log --oneline -1
c0ffee4 Add the frobnicator
```

Six digits is a 2^24 search, over before the progress meter draws.
The message, tree, author and date are untouched; the padding is real
but invisible:

```
$ git cat-file commit HEAD | tail -1 | cat -A
                                               ^I^I^I^I ^I^I^I^I^I^I  ^I ^I   ^I^I     ^I^I
```

Longer prefixes take real time and keep a progress line updating while
they sweep. `--dry-run` runs the same search but leaves `HEAD` alone,
which is the way to feel out a prefix length before adopting it:

```
$ git brew --dry-run c0ffeebad
searching for prefix 'c0ffeebad': ~2^36 of 2^48 candidates expected
  11.25 GH/s    0.01% swept      2 s elapsed
c0ffee4ffc212a7760b30c407c87ad2ce16807c8 -> c0ffeebadf60ed0751fa40c1f18e86d9b2d5f724 (dry run, nothing written)
```

## What it refuses to do

- **Amend a pushed commit.** If `HEAD` is contained in any
  remote-tracking ref, it stops rather than rewrite published history.
  `--force` overrides this, and only this.
- **Touch a signed commit.** A `gpgsig` signature covers the whole
  object, so appending to the message would break verification. There
  is no override; re-sign after amending instead.
- **Run on a SHA-256 repository.** SHA-1 only, for now.

Nothing is written until the winning object has been re-hashed by the
tool's own SHA-1, matched against the prefix, and confirmed identical
to what `git hash-object` computes. The GPU is a search oracle and
nothing it returns is trusted without reverification.

## How fast

Measured on a GB10 (48 SMs) at a steady ~11 GH/s of single-block SHA-1
with early rejection. Times are the mean to find a prefix of the given
length (2^(4n) candidates for n hex digits):

| Prefix | Candidates | Time |
|--------|-----------|-------------------|
| 7 hex  | 2^28      | instant (this is the abbreviated hash GitHub shows) |
| 8 hex  | 2^32      | under half a second |
| 10 hex | 2^40      | ~1.7 minutes |
| 11 hex | 2^44      | ~27 minutes |
| 12 hex | 2^48      | ~7 hours |

12 hex digits is right at the edge: the search space is 2^48 and so is
the expected work, so a single sweep finds it only about two times in
three.

The kernel is bound by the SHA-1 round dependency chain, not occupancy
or memory, so the throughput is flat across launch shapes - see the
comment in `native/backend/cuda/kernel.cu`.

## Development

The native core carries its own test suite, runnable without cargo:

```
cmake -S native -B native/build -G Ninja
cmake --build native/build
ctest --test-dir native/build --output-on-failure
```

Those cover the SHA-1 vectors, the candidate encoder, and CPU-vs-GPU
parity through the real search entry point. The Rust side adds commit
surgery, the git plumbing wrappers, and end-to-end tests against scratch
repositories:

```
cargo test
```

Tests that reach the GPU search skip themselves when
`GIT_BREW_SKIP_GPU_TESTS` is set, which is how CI runs the suite on
GPU-less runners; the native equivalent is `ctest -E gpu_parity`. With
the variable unset, everything runs.

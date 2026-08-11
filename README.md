# git-brew

Brute-force a chosen hex prefix onto a git commit hash, on the GPU.
Point it at `HEAD`, name a prefix, and it rewrites the commit so its
SHA-1 starts with your digits - by appending invisible whitespace to
the message, so nothing you can see changes.

```
$ git brew c0ffee
searching for prefix 'c0ffee': ~2^24 of 2^48 candidates expected
   11.02 GH/s    0.31% swept      1 s elapsed
7a3f9c1d... -> c0ffee2b8d4e6a1f0c9b7d5e3a2f8c6b4d1e0a9f
```

The commit message, your tree, your author and date all stay exactly
as they were. The only change is a run of trailing spaces and tabs on
the last line, which every git UI ignores.

## Why a GPU, and why CUDA

The honest answer is in `DESIGN.md`. The short version: SHA-1's round
functions are three-input bitwise ops that collapse to a single
`lop3.b32` each, and that plus Nsight Compute made CUDA the right first
backend for a project whose entire point is hashes per second. A
portable `wgpu` backend is a follow-up, not a promise.

It leans on one trick worth knowing: SHA-1 processes 64-byte blocks in
sequence, so if every byte that changes between candidates lives in the
*final* block, the hash of everything before it is computed once on the
host and the GPU does a single compression per candidate. The tool
sizes a run of trailing whitespace to land exactly on that final block
boundary, which is what makes it fast instead of useless.

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
three. Past 12 is a different project.

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

Everything from the Rust FFI smoke test onward needs a visible GPU.

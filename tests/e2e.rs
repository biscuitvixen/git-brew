//! End-to-end tests: the built binary against real scratch repositories.
//!
//! These run the actual `git-brew` executable (cargo exposes its path as
//! CARGO_BIN_EXE_git-brew) so argument parsing, the refusal paths, the
//! GPU search, and the ref update are all exercised exactly as a user
//! would hit them. Prefixes are kept to two hex digits: a 2^8 expected
//! search is milliseconds even on the CPU-side overhead alone.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

const BIN: &str = env!("CARGO_BIN_EXE_git-brew");

/// Mirrors crate::cuda::skip_gpu_tests (integration tests are a separate
/// crate): tests that reach the GPU search early-return when
/// GIT_BREW_SKIP_GPU_TESTS is set. The refusal and usage-error paths
/// never touch the device and always run.
fn skip_gpu_tests() -> bool {
    if std::env::var_os("GIT_BREW_SKIP_GPU_TESTS").is_some() {
        eprintln!("skipping: GIT_BREW_SKIP_GPU_TESTS is set");
        return true;
    }
    false
}

fn scratch_repo(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("git-brew-e2e-{}-{name}", std::process::id()));
    if dir.exists() {
        fs::remove_dir_all(&dir).expect("stale scratch repo must be removable");
    }
    fs::create_dir_all(&dir).expect("scratch repo dir must be creatable");
    git(&dir, &["init", "-q"]);
    git(&dir, &["config", "user.name", "Test Committer"]);
    git(&dir, &["config", "user.email", "committer@example.com"]);
    commit(&dir, "End-to-end fixture commit");
    dir
}

fn git(dir: &Path, args: &[&str]) -> String {
    let output = Command::new("git")
        .args(args)
        .current_dir(dir)
        .env("GIT_AUTHOR_NAME", "Test Author")
        .env("GIT_AUTHOR_EMAIL", "author@example.com")
        .env("GIT_AUTHOR_DATE", "1700000000 +0000")
        .env("GIT_COMMITTER_DATE", "1700000000 +0000")
        .output()
        .expect("git must be runnable");
    assert!(
        output.status.success(),
        "git {args:?} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout)
        .trim_end()
        .to_string()
}

fn commit(dir: &Path, message: &str) {
    git(dir, &["commit", "-q", "--allow-empty", "-m", message]);
}

fn run_tool(dir: &Path, args: &[&str]) -> Output {
    Command::new(BIN)
        .args(args)
        .current_dir(dir)
        .stdin(Stdio::null())
        .output()
        .expect("git-brew must be runnable")
}

fn stderr_of(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[test]
fn a_two_digit_prefix_lands_on_head_and_the_repo_stays_healthy() {
    if skip_gpu_tests() {
        return;
    }
    let repo = scratch_repo("happy-path");
    let before = git(&repo, &["rev-parse", "HEAD"]);
    let output = run_tool(&repo, &["2b"]);
    assert!(
        output.status.success(),
        "tool failed: {}",
        stderr_of(&output)
    );
    let after = git(&repo, &["rev-parse", "HEAD"]);
    assert!(
        after.starts_with("2b"),
        "HEAD {after} does not start with the prefix"
    );
    assert_ne!(before, after, "HEAD must have moved");
    let fsck = Command::new("git")
        .args(["fsck", "--strict"])
        .current_dir(&repo)
        .output()
        .expect("git fsck must run");
    assert!(
        fsck.status.success(),
        "git fsck rejects the rewritten repo: {}",
        String::from_utf8_lossy(&fsck.stderr)
    );
    let message = git(&repo, &["log", "-1", "--format=%B"]);
    assert_eq!(
        message.trim_end(),
        "End-to-end fixture commit",
        "the visible message must be unchanged"
    );
}

#[test]
fn dry_run_reports_the_hash_without_moving_head() {
    if skip_gpu_tests() {
        return;
    }
    let repo = scratch_repo("dry-run");
    let before = git(&repo, &["rev-parse", "HEAD"]);
    let output = run_tool(&repo, &["--dry-run", "3c"]);
    assert!(
        output.status.success(),
        "dry run failed: {}",
        stderr_of(&output)
    );
    assert_eq!(
        git(&repo, &["rev-parse", "HEAD"]),
        before,
        "dry run must not move HEAD"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let reported = stdout
        .split_whitespace()
        .nth(2)
        .expect("dry run prints 'old -> new ...'");
    assert!(
        reported.starts_with("3c"),
        "reported hash {reported} does not start with the prefix"
    );
}

#[test]
fn the_configured_prefix_is_used_when_no_argument_is_given() {
    if skip_gpu_tests() {
        return;
    }
    let repo = scratch_repo("config-prefix");
    git(&repo, &["config", "brew.prefix", "4d"]);
    let output = run_tool(&repo, &[]);
    assert!(
        output.status.success(),
        "tool failed: {}",
        stderr_of(&output)
    );
    let after = git(&repo, &["rev-parse", "HEAD"]);
    assert!(after.starts_with("4d"), "HEAD {after} ignores brew.prefix");
}

#[test]
fn a_signed_commit_is_refused() {
    let repo = scratch_repo("signed");
    // A commit object with a syntactically plausible (never verifiable)
    // signature header, written via plumbing so no real key is needed.
    let tree = git(&repo, &["rev-parse", "HEAD^{tree}"]);
    let content = format!(
        "tree {tree}\n\
         author Test Author <author@example.com> 1700000000 +0000\n\
         committer Test Committer <committer@example.com> 1700000000 +0000\n\
         gpgsig -----BEGIN PGP SIGNATURE-----\n \n -----END PGP SIGNATURE-----\n\
         \n\
         Signed fixture\n"
    );
    let oid = {
        let mut child = Command::new("git")
            .args(["hash-object", "-t", "commit", "-w", "--stdin"])
            .current_dir(&repo)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("git hash-object must spawn");
        use std::io::Write as _;
        child
            .stdin
            .take()
            .expect("stdin piped")
            .write_all(content.as_bytes())
            .expect("write commit content");
        let out = child.wait_with_output().expect("hash-object output");
        assert!(out.status.success(), "hash-object failed");
        String::from_utf8_lossy(&out.stdout).trim_end().to_string()
    };
    git(&repo, &["update-ref", "HEAD", &oid]);
    let output = run_tool(&repo, &["5e"]);
    assert!(!output.status.success(), "a signed commit must be refused");
    assert!(
        stderr_of(&output).contains("signed"),
        "refusal must explain the signature problem, got: {}",
        stderr_of(&output)
    );
}

#[test]
fn a_pushed_commit_is_refused_unless_forced() {
    let repo = scratch_repo("pushed");
    let head = git(&repo, &["rev-parse", "HEAD"]);
    git(&repo, &["update-ref", "refs/remotes/origin/main", &head]);
    let refused = run_tool(&repo, &["6f"]);
    assert!(!refused.status.success(), "a pushed commit must be refused");
    assert!(
        stderr_of(&refused).contains("origin/main"),
        "refusal must name the remote ref, got: {}",
        stderr_of(&refused)
    );
    // The refusal above is CPU-only; the forced run below searches.
    if skip_gpu_tests() {
        return;
    }
    let forced = run_tool(&repo, &["--force", "6f"]);
    assert!(
        forced.status.success(),
        "--force must override the pushed check: {}",
        stderr_of(&forced)
    );
    assert!(
        git(&repo, &["rev-parse", "HEAD"]).starts_with("6f"),
        "forced run must still land the prefix"
    );
}

#[test]
fn rerunning_replaces_padding_instead_of_growing_the_object() {
    if skip_gpu_tests() {
        return;
    }
    let repo = scratch_repo("idempotent");
    for prefix in ["a1", "b2"] {
        let output = run_tool(&repo, &[prefix]);
        assert!(
            output.status.success(),
            "run '{prefix}' failed: {}",
            stderr_of(&output)
        );
    }
    let size: usize = git(&repo, &["cat-file", "-s", "HEAD"])
        .parse()
        .expect("size parses");
    // One padded object: base content (~215 bytes) + filler + 48. Two runs
    // that accumulated padding would exceed this comfortably.
    assert!(
        size < 220 + 128 + 48,
        "object size {size} suggests padding accumulated across runs"
    );
}

#[test]
fn a_bad_prefix_is_a_usage_error() {
    let repo = scratch_repo("bad-prefix");
    let output = run_tool(&repo, &["not-hex"]);
    assert_eq!(output.status.code(), Some(2), "usage errors exit 2");
    let output = run_tool(&repo, &["0123456789abcd"]);
    assert_eq!(output.status.code(), Some(2), "a 14-digit prefix exits 2");
}

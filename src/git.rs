//! Thin wrappers around git plumbing commands.
//!
//! Everything goes through the `git` binary rather than a library binding:
//! object hashing and ref updates then carry git's own validation, and the
//! only dependency is the git already on the machine. Commit content is
//! handled as `Vec<u8>` throughout - author names and messages are not
//! guaranteed UTF-8. Only oids and config values, which git defines as
//! ASCII, are returned as `String`.
//!
//! This module knows nothing about hashing or padding; it moves bytes and
//! refs. All functions take the repository directory explicitly so tests
//! can drive scratch repositories.

use std::fmt;
use std::io::Write as _;
use std::path::Path;
use std::process::{Command, Stdio};

#[derive(Debug)]
pub struct GitError {
    /// The subcommand that failed, for the error message.
    command: String,
    /// Captured stderr, or the IO error that prevented the run.
    detail: String,
}

impl fmt::Display for GitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "git {} failed: {}", self.command, self.detail.trim_end())
    }
}

impl std::error::Error for GitError {}

/// Run a git subcommand, optionally feeding stdin, and return its stdout.
fn run(repo: &Path, args: &[&str], stdin: Option<&[u8]>) -> Result<Vec<u8>, GitError> {
    let command = args.join(" ");
    let err = |detail: String| GitError {
        command: command.clone(),
        detail,
    };

    let mut child = Command::new("git")
        .args(args)
        .current_dir(repo)
        .stdin(if stdin.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| err(format!("could not run git: {e}")))?;
    if let Some(bytes) = stdin {
        // Writing everything before reading cannot deadlock here: every
        // stdin-fed command (hash-object) emits far less than a pipe buffer.
        child
            .stdin
            .take()
            .expect("stdin was requested piped")
            .write_all(bytes)
            .map_err(|e| err(format!("could not write to git stdin: {e}")))?;
    }
    let output = child
        .wait_with_output()
        .map_err(|e| err(format!("could not collect git output: {e}")))?;
    if !output.status.success() {
        return Err(err(String::from_utf8_lossy(&output.stderr).into_owned()));
    }
    Ok(output.stdout)
}

/// Run a git subcommand whose output is a single ASCII line.
fn run_line(repo: &Path, args: &[&str]) -> Result<String, GitError> {
    let out = run(repo, args, None)?;
    Ok(String::from_utf8_lossy(&out).trim_end().to_string())
}

/// The commit oid HEAD points at. Fails on an unborn branch.
pub fn head_commit_oid(repo: &Path) -> Result<String, GitError> {
    run_line(repo, &["rev-parse", "--verify", "HEAD^{commit}"])
}

/// The repository's object format: "sha1" or "sha256".
pub fn object_format(repo: &Path) -> Result<String, GitError> {
    run_line(repo, &["rev-parse", "--show-object-format"])
}

/// Raw content of a commit object, wrapper already stripped by git.
pub fn cat_commit(repo: &Path, oid: &str) -> Result<Vec<u8>, GitError> {
    run(repo, &["cat-file", "commit", oid], None)
}

/// Hash commit content the way git would; store the object when `write`.
/// Returns the oid git computed, which the caller cross-checks against its
/// own prediction.
pub fn hash_object(repo: &Path, content: &[u8], write: bool) -> Result<String, GitError> {
    let mut args = vec!["hash-object", "-t", "commit", "--stdin"];
    if write {
        args.insert(1, "-w");
    }
    let out = run(repo, &args, Some(content))?;
    Ok(String::from_utf8_lossy(&out).trim_end().to_string())
}

/// Point HEAD at `new_oid`, but only if it still points at `old_oid`: the
/// old-value argument gives compare-and-swap semantics against concurrent
/// modification, and works the same on a branch or a detached HEAD.
pub fn update_head(
    repo: &Path,
    new_oid: &str,
    old_oid: &str,
    reflog_message: &str,
) -> Result<(), GitError> {
    run(
        repo,
        &["update-ref", "-m", reflog_message, "HEAD", new_oid, old_oid],
        None,
    )?;
    Ok(())
}

/// Short names of remote-tracking refs whose history contains `oid`. A
/// non-empty result means the commit has been pushed somewhere and amending
/// it would rewrite published history.
pub fn remote_refs_containing(repo: &Path, oid: &str) -> Result<Vec<String>, GitError> {
    let listing = run(
        repo,
        &[
            "for-each-ref",
            "--format=%(refname:short) %(objectname)",
            "refs/remotes",
        ],
        None,
    )?;
    let mut containing = Vec::new();
    for line in String::from_utf8_lossy(&listing).lines() {
        let Some((name, tip)) = line.split_once(' ') else {
            continue;
        };
        // Exit 0 means ancestor, 1 means not; anything else is a real error.
        let status = Command::new("git")
            .args(["merge-base", "--is-ancestor", oid, tip])
            .current_dir(repo)
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .output()
            .map_err(|e| GitError {
                command: "merge-base --is-ancestor".to_string(),
                detail: format!("could not run git: {e}"),
            })?;
        match status.status.code() {
            Some(0) => containing.push(name.to_string()),
            Some(1) => {}
            _ => {
                return Err(GitError {
                    command: "merge-base --is-ancestor".to_string(),
                    detail: String::from_utf8_lossy(&status.stderr).into_owned(),
                })
            }
        }
    }
    Ok(containing)
}

/// A config value, or None when the key is unset.
pub fn config(repo: &Path, key: &str) -> Result<Option<String>, GitError> {
    let output = Command::new("git")
        .args(["config", "--get", key])
        .current_dir(repo)
        .output()
        .map_err(|e| GitError {
            command: format!("config --get {key}"),
            detail: format!("could not run git: {e}"),
        })?;
    match output.status.code() {
        Some(0) => Ok(Some(
            String::from_utf8_lossy(&output.stdout)
                .trim_end()
                .to_string(),
        )),
        Some(1) => Ok(None),
        _ => Err(GitError {
            command: format!("config --get {key}"),
            detail: String::from_utf8_lossy(&output.stderr).into_owned(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sha1;
    use std::fs;
    use std::path::PathBuf;

    /// A fresh scratch repository with one deterministic commit.
    fn scratch_repo(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("git-brew-test-{}-{name}", std::process::id()));
        if dir.exists() {
            fs::remove_dir_all(&dir).expect("stale scratch repo must be removable");
        }
        fs::create_dir_all(&dir).expect("scratch repo dir must be creatable");
        git_in(&dir, &["init", "-q"]);
        git_in(&dir, &["config", "user.name", "Test Committer"]);
        git_in(&dir, &["config", "user.email", "committer@example.com"]);
        commit_in(&dir, "Scratch fixture commit");
        dir
    }

    fn git_in(dir: &Path, args: &[&str]) {
        let status = Command::new("git")
            .args(args)
            .current_dir(dir)
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .status()
            .expect("git must be runnable");
        assert!(status.success(), "git {args:?} failed in {dir:?}");
    }

    fn commit_in(dir: &Path, message: &str) {
        let status = Command::new("git")
            .args(["commit", "-q", "--allow-empty", "-m", message])
            .current_dir(dir)
            .env("GIT_AUTHOR_NAME", "Test Author")
            .env("GIT_AUTHOR_EMAIL", "author@example.com")
            .env("GIT_AUTHOR_DATE", "1700000000 +0000")
            .env("GIT_COMMITTER_DATE", "1700000000 +0000")
            .stdout(Stdio::null())
            .status()
            .expect("git commit must be runnable");
        assert!(status.success(), "commit failed in {dir:?}");
    }

    #[test]
    fn cat_commit_roundtrips_through_the_wrapper_to_the_oid() {
        let repo = scratch_repo("roundtrip");
        let oid = head_commit_oid(&repo).expect("HEAD must resolve");
        let content = cat_commit(&repo, &oid).expect("cat-file must succeed");
        let mut object = format!("commit {}\0", content.len()).into_bytes();
        object.extend_from_slice(&content);
        assert_eq!(
            sha1::to_hex(&sha1::digest(&object)),
            oid,
            "re-wrapped content must hash back to the oid git reported"
        );
    }

    #[test]
    fn hash_object_agrees_with_the_host_sha1() {
        let repo = scratch_repo("hash-object");
        let oid = head_commit_oid(&repo).expect("HEAD must resolve");
        let content = cat_commit(&repo, &oid).expect("cat-file must succeed");
        let hashed = hash_object(&repo, &content, false).expect("hash-object dry run");
        assert_eq!(
            hashed, oid,
            "hash-object must reproduce the oid without writing"
        );
    }

    #[test]
    fn update_head_moves_head_only_from_the_expected_old_oid() {
        let repo = scratch_repo("update-head");
        let old = head_commit_oid(&repo).expect("HEAD must resolve");
        let mut content = cat_commit(&repo, &old).expect("cat-file must succeed");
        content.extend_from_slice(b" ");
        let new = hash_object(&repo, &content, true).expect("hash-object -w");
        update_head(&repo, &new, &old, "git-brew test").expect("update-ref");
        assert_eq!(
            head_commit_oid(&repo).expect("HEAD must resolve"),
            new,
            "HEAD must point at the rewritten commit"
        );
        update_head(&repo, &old, &old, "git-brew test stale")
            .expect_err("a stale old-oid must be rejected by the CAS check");
    }

    #[test]
    fn remote_refs_report_exactly_the_refs_containing_head() {
        let repo = scratch_repo("remotes");
        let first = head_commit_oid(&repo).expect("HEAD must resolve");
        assert_eq!(
            remote_refs_containing(&repo, &first).expect("check must run"),
            Vec::<String>::new(),
            "no remote refs exist yet"
        );
        git_in(&repo, &["update-ref", "refs/remotes/origin/main", &first]);
        assert_eq!(
            remote_refs_containing(&repo, &first).expect("check must run"),
            vec!["origin/main".to_string()],
            "the tracking ref contains the pushed commit"
        );
        commit_in(&repo, "Unpushed follow-up");
        let second = head_commit_oid(&repo).expect("HEAD must resolve");
        assert_eq!(
            remote_refs_containing(&repo, &second).expect("check must run"),
            Vec::<String>::new(),
            "the new commit is not on any remote ref"
        );
    }

    #[test]
    fn config_distinguishes_unset_from_set() {
        let repo = scratch_repo("config");
        assert_eq!(
            config(&repo, "brew.prefix").expect("config must run"),
            None,
            "brew.prefix starts unset"
        );
        git_in(&repo, &["config", "brew.prefix", "c0ffee"]);
        assert_eq!(
            config(&repo, "brew.prefix").expect("config must run"),
            Some("c0ffee".to_string()),
            "brew.prefix reads back"
        );
    }

    #[test]
    fn object_format_is_sha1_in_a_default_repo() {
        let repo = scratch_repo("object-format");
        assert_eq!(
            object_format(&repo).expect("rev-parse must run"),
            "sha1",
            "default repositories use SHA-1"
        );
    }
}

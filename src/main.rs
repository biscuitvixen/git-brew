//! git-brew: brute-force a chosen hex prefix onto HEAD's commit hash.
//!
//! The flow is deliberately paranoid: the commit object must round-trip
//! through the host SHA-1 before any surgery, and the winning object must
//! hash to the same oid under the host SHA-1 and under `git hash-object`
//! before HEAD moves. The GPU is only ever a search oracle; nothing it
//! returns is trusted without reverification.

mod backend;
mod commit;
mod cuda;
mod git;
// Not yet reachable from main; the tests are its only caller.
#[cfg_attr(not(test), expect(dead_code))]
mod host;
mod search;
mod sha1;

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

const USAGE: &str = "\
usage: git-brew [--dry-run] [--force] [<prefix>]

Rewrite HEAD so its hash starts with <prefix> (1-12 hex digits), by
appending invisible whitespace to the commit message. Reads the prefix
from `git config brew.prefix` when the argument is omitted.

options:
  --dry-run   search and verify, print the would-be hash, write nothing
  --force     amend even when HEAD is already on a remote-tracking ref
  --help      show this text
  --version   show the version";

struct Args {
    prefix: Option<String>,
    dry_run: bool,
    force: bool,
}

enum ArgsOutcome {
    Run(Args),
    /// --help or --version: text already printed, exit cleanly.
    Done,
}

fn parse_args(argv: impl Iterator<Item = String>) -> Result<ArgsOutcome, String> {
    let mut args = Args {
        prefix: None,
        dry_run: false,
        force: false,
    };
    for arg in argv {
        match arg.as_str() {
            "--dry-run" => args.dry_run = true,
            "--force" => args.force = true,
            "--help" | "-h" => {
                println!("{USAGE}");
                return Ok(ArgsOutcome::Done);
            }
            "--version" => {
                println!("git-brew {}", env!("CARGO_PKG_VERSION"));
                return Ok(ArgsOutcome::Done);
            }
            s if s.starts_with('-') => return Err(format!("unknown option '{s}'")),
            s => {
                if args.prefix.replace(s.to_string()).is_some() {
                    return Err("more than one prefix argument".to_string());
                }
            }
        }
    }
    Ok(ArgsOutcome::Run(args))
}

/// A usage mistake (exit 2) or a runtime failure/refusal (exit 1).
enum ToolError {
    Usage(String),
    Fail(String),
}

impl From<String> for ToolError {
    fn from(msg: String) -> Self {
        ToolError::Fail(msg)
    }
}

/// Lowercase and validate the prefix, one nibble value per hex digit.
fn parse_prefix(prefix: &str) -> Result<Vec<u8>, ToolError> {
    if prefix.is_empty() || prefix.len() > backend::MAX_PREFIX_NIBBLES {
        return Err(ToolError::Usage(format!(
            "prefix '{prefix}' must be 1 to {} hex digits",
            backend::MAX_PREFIX_NIBBLES
        )));
    }
    prefix
        .chars()
        .map(|c| {
            c.to_digit(16).map(|d| d as u8).ok_or_else(|| {
                ToolError::Usage(format!("prefix '{prefix}' contains non-hex '{c}'"))
            })
        })
        .collect()
}

fn run_tool(repo: &Path, args: &Args) -> Result<(), ToolError> {
    let prefix = match &args.prefix {
        Some(p) => p.to_ascii_lowercase(),
        None => git::config(repo, "brew.prefix")
            .map_err(|e| e.to_string())?
            .ok_or(ToolError::Usage(
                "no prefix argument and no brew.prefix in git config".to_string(),
            ))?,
    };
    let nibbles = parse_prefix(&prefix)?;

    let format = git::object_format(repo).map_err(|e| e.to_string())?;
    if format != "sha1" {
        return Err(
            format!("repository uses the {format} object format; only sha1 is supported").into(),
        );
    }
    let old_oid = git::head_commit_oid(repo)
        .map_err(|e| format!("cannot resolve HEAD (unborn branch or not a repository?): {e}"))?;
    let content = git::cat_commit(repo, &old_oid).map_err(|e| e.to_string())?;

    // Round-trip check: re-wrapping what git handed over must reproduce the
    // oid, or some assumption about the object encoding is wrong and no
    // surgery can be trusted.
    let mut object = format!("commit {}\0", content.len()).into_bytes();
    object.extend_from_slice(&content);
    let round_trip = sha1::to_hex(&sha1::digest(&object));
    if round_trip != old_oid {
        return Err(format!(
            "HEAD did not round-trip: git says {old_oid}, host SHA-1 says {round_trip}"
        )
        .into());
    }

    let (headers, _) = commit::split(&content).map_err(|e| e.to_string())?;
    if commit::has_gpgsig(headers) {
        return Err(
            "HEAD is signed; padding the message would break the signature. \
             Amend without a signature first (this check has no override)"
                .to_string()
                .into(),
        );
    }
    let pushed = git::remote_refs_containing(repo, &old_oid).map_err(|e| e.to_string())?;
    if !pushed.is_empty() && !args.force {
        return Err(format!(
            "HEAD is already on {} and amending would rewrite pushed history \
             (use --force to override)",
            pushed.join(", ")
        )
        .into());
    }

    let padded = commit::prepare(&content).map_err(|e| e.to_string())?;
    let mut midstate = sha1::INIT_STATE;
    sha1::update_blocks(&mut midstate, padded.midstate_prefix());
    let tail = padded.tail();

    let interrupt = Arc::new(AtomicBool::new(false));
    let handler_flag = Arc::clone(&interrupt);
    ctrlc::set_handler(move || handler_flag.store(true, Ordering::Relaxed))
        .map_err(|e| format!("could not install the Ctrl-C handler: {e}"))?;

    let mut searcher = cuda::Searcher::new().map_err(|e| e.to_string())?;
    eprintln!(
        "searching for prefix '{prefix}': ~2^{} of 2^{} candidates expected",
        4 * nibbles.len(),
        search::SPACE_BITS
    );
    let outcome = search::run(
        &mut searcher,
        &midstate,
        &tail,
        &nibbles,
        0..1 << search::SPACE_BITS,
        &interrupt,
        &mut |p| {
            eprint!(
                "\r{:>7.2} GH/s  {:>6.2}% swept  {:>5.0} s elapsed ",
                p.rate / 1e9,
                p.done as f64 / p.total as f64 * 100.0,
                p.elapsed.as_secs_f64()
            );
        },
    )
    .map_err(|e| e.to_string())?;
    eprintln!();

    let candidate = match outcome {
        search::Outcome::Found(c) => c,
        search::Outcome::Interrupted => return Err("interrupted".to_string().into()),
        search::Outcome::Exhausted => {
            return Err("the whole 48-bit pad space was swept without a hit; \
                        amend the commit (changing its timestamp) and re-run"
                .to_string()
                .into())
        }
    };

    // Reverify the winner twice over before touching anything: the host
    // SHA-1 must see the prefix, and git must agree on the digest.
    let new_object = padded.object_for(candidate);
    let new_oid = sha1::to_hex(&sha1::digest(&new_object));
    if !new_oid.starts_with(&prefix) {
        return Err(format!(
            "internal error: GPU candidate {candidate:#x} hashes to {new_oid}, \
             which does not start with '{prefix}'"
        )
        .into());
    }
    let nul = new_object
        .iter()
        .position(|&b| b == 0)
        .expect("wrapper has a NUL");
    let new_content = &new_object[nul + 1..];
    let git_oid = git::hash_object(repo, new_content, !args.dry_run).map_err(|e| e.to_string())?;
    if git_oid != new_oid {
        return Err(format!(
            "internal error: git hashed the object to {git_oid}, host SHA-1 to {new_oid}; \
             nothing was updated"
        )
        .into());
    }

    if args.dry_run {
        println!("{old_oid} -> {new_oid} (dry run, nothing written)");
        return Ok(());
    }
    let reflog = format!("git-brew: {} -> {}", &old_oid[..7], &new_oid[..7]);
    git::update_head(repo, &new_oid, &old_oid, &reflog).map_err(|e| e.to_string())?;
    println!("{old_oid} -> {new_oid}");
    Ok(())
}

fn main() -> ExitCode {
    let args = match parse_args(std::env::args().skip(1)) {
        Ok(ArgsOutcome::Run(args)) => args,
        Ok(ArgsOutcome::Done) => return ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("git-brew: {msg}\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    match run_tool(&PathBuf::from("."), &args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(ToolError::Usage(msg)) => {
            eprintln!("git-brew: {msg}\n{USAGE}");
            ExitCode::from(2)
        }
        Err(ToolError::Fail(msg)) => {
            eprintln!("git-brew: {msg}");
            ExitCode::FAILURE
        }
    }
}

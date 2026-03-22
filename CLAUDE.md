# CLAUDE.md - Bot Instructions for hw-jjg-bot

## Project Structure

This project uses **two separate jj-git repos**:

1. **App repo** (`/` — project root): Contains the application source code.
2. **Bot session repo** (`/.claude/`): Contains Claude Code session data.

Both repos are managed with `jj` (Jujutsu), which coexists with git.

## Repo Paths (relative from project root)

- App repo: `.` (project root)
- Bot session repo: `.claude`
  (symlink from `~/.claude/projects/<path-to-project-root>/.claude`)

## Committing

Use `-R` (`--repository`) at the end to target the correct repo. Use
relative paths to reduce noise. Putting `-R` last keeps the verb/action
visible at the start of the command.

### App repo
```
jj commit -m "title" -m "body" -R .
```

### Bot session repo
```
jj commit -m "title" -m "body" -R .claude
```

## jj Basics

- `jj st -R .` / `jj st -R .claude` — show working copy status
- `jj log -R .` / `jj log -R .claude` — show commit log
- `jj commit -m "title" -m "body" -R <repo>` — finalize working copy into a commit
- `jj describe -m "title" -m "body" -R <repo>` — set description without committing
- In jj, the working copy (@) is always a mutable commit being edited.
  `jj commit` finalizes it and creates a new empty working copy on top.
- The `.claude` repo always has uncommitted changes during an active
  session because session data updates continuously.

## Pre-commit Requirements

### User approval

Never execute commit commands without the user's explicit approval.
Present the full commands for review first; only run them after the
user confirms.

### Versioning

Every change must start with a version bump. See
[Versioning during development](notes/README.md#versioning-during-development)
for details. Get user approval on single-step vs multi-step before starting.

### Pre-commit checklist

Before proposing a commit, run all of the following and fix any issues:

1. `cargo fmt`
2. `cargo clippy`
3. `cargo test`
4. `cargo install --path .` (if applicable)
5. Retest after install

## Session End Workflows

When the user asks to "commit both repos" or says they're done, commit
both repos. Use the **same title** for both commits so they're easy to
correlate. The body can differ: the app repo body should summarize code
changes; the bot session repo body should note what was done in the
session.

```
jj commit -m "shared title" -m "app body" -R .
jj commit -m "shared title" -m "session body" -R .claude
```

When the user also asks to push, advance the `main` bookmarks to the
new commits first, then push.

```
jj bookmark set main -r @- -R .
jj bookmark set main -r @- -R .claude
jj git push -R .
jj git push -R .claude
```

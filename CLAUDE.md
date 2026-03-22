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

## Working Directory

Prefer staying in the project root. Use `-R` flags or absolute paths
to target other directories rather than `cd`. If `cd` seems necessary,
discuss with the user first — losing track of cwd causes subtle
command failures downstream.

## Committing

Use `-R` (`--repository`) at the end to target the correct repo. Use
relative paths to reduce noise. Putting `-R` last keeps the verb/action
visible at the start of the command.

### App repo
```
jj commit -m \
"title" \
-m "body

ochid: /.claude/<changeID>" \
-R .
```

### Bot session repo
```
jj commit -m \
"title" \
-m "body

ochid: /<changeID>" \
-R .claude
```

## jj Basics

- `jj st -R .` / `jj st -R .claude` — show working copy status
- `jj log -R .` / `jj log -R .claude` — show commit log
- `jj commit -m "title" -m "body" -R <repo>` — finalize working copy into a commit
- `jj describe -m "title" -m "body" -R <repo>` — set description without committing
- `jj git push --bookmark <name> -R <repo>` — push a bookmark (no
  `--allow-new` flag; jj pushes new bookmarks without special flags)
- In jj, the working copy (@) is always a mutable commit being edited.
  `jj commit` finalizes it and creates a new empty working copy on top.
- The `.claude` repo always has uncommitted changes during an active
  session because session data updates continuously.

## Commit Message Style

Use [Conventional Commits](https://www.conventionalcommits.org/) with
a version suffix:

```
<type>: <short description> (<version>)
```

- **Title**: target ~50 chars, short summary of *what* changed.
  Include the version. Common types: `feat`, `fix`, `refactor`,
  `test`, `docs`, `chore`.
- **Body**: expand on *what* if needed, plus short *why* and *how*.
- Examples:
  - `feat: add fix-ochid subcommand (0.22.0)`
  - `fix: fix-ochid prefix bug (0.22.1)`
  - `refactor: deduplicate common CLI flags (0.21.1)`

## Pre-commit Requirements

### User approval

Never execute commit, squash, push, or finalize commands without the
user's explicit approval. Present changes for review first; only run
them after the user confirms. This applies to late changes too —
pause for review before squashing into an existing commit.

### Notes references

Multiple references must be separated: `[2],[3]` not `[2,3]` or `[2][3]`.
See [Todo format](notes/README.md#todo-format) for details.

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
6. Update `notes/todo.md` — add to `## Done` if completing a task
7. Update `notes/chores-*.md` — add a subsection describing the change
8. Update `notes/README.md` — if functionality changed (new flags,
   new subcommands, changed behavior)

## ochid Trailers

Every commit body must include an `ochid:` trailer pointing to the
counterpart commit in the other repo. The value is a workspace-root-relative
path followed by the changeID:

- App repo commits point to `.claude`: `ochid: /.claude/<changeID>`
- Bot session commits point to app repo: `ochid: /<changeID>`

Use `vc-x1 chid -R .,.claude -L` to get both changeIDs (first line
is app repo, second is `.claude`).

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

After both commits, **pause and ask for user approval** before
proceeding to bookmark/push/finalize.

When the user also asks to push:

1. Advance the current bookmark on both repos
2. Push the app repo (do **not** push `.claude` here)
3. **Wait for user approval** before proceeding to finalize
4. Finalize `.claude` repo (very last action — see below)

Replace `<bookmark>` with the active bookmark (e.g. `main`,
`dev-0.14.0`).

```
jj bookmark set <bookmark> -r @- -R .
jj bookmark set <bookmark> -r @- -R .claude
jj git push --bookmark <bookmark> -R .
```

### Late changes after push

If changes are made to the app repo after it has been pushed (e.g.
updating CLAUDE.md or memory), the commit is now immutable. Use
`--ignore-immutable` to squash the changes in, then re-push:

```
jj squash --ignore-immutable -R .
jj bookmark set <bookmark> -r @- -R .
jj git push --bookmark <bookmark> -R .
```

### Finalize the .claude repo

The **very last action** in a session is to finalize the `.claude` repo.
This squashes the working copy into the session commit and pushes. The
delay gives a safety margin against any pending writes. Always use a
short relative path for `--repo`.

**Nothing should happen after finalize** — no memory writes, no tool
calls, no additional output. If any work is done after finalize, run
finalize again so the trailing writes are captured.

`--bookmark` is required — use the active bookmark for the session.

```
vc-x1 finalize --repo .claude --bookmark <bookmark> --delay 10 --detach --push
```

Do **not** echo or restate the finalize output — the Bash tool
already displays it. Any trailing text output creates writes that
miss the finalize squash window.

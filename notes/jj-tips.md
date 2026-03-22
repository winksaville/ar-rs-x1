# Jujutsu (jj) Tips

See [Steve Klabnik](https://github.com/steveklabnik)
[Jujutsu-tutorial](https://steveklabnik.github.io/jujutsu-tutorial)
and [jj docs](https://docs.jj-vcs.dev/latest/).

## Initial Commit for a repo

Create create directory add files.

Minimal commands to push 

```
jj git init .
jj describe
jj git remote add origin git@github.com:winksaville/vc-template-x1
jj bookmark create main -r @
jj bookmark track main --remote=origin
jj git push
```

## Push a change to main

Assuming that this is to be push to main you
set the bookmark to the appropriate commit and
then just push:

```
jj bookmark set main -r @
jj git push
```

Complete example:
```
wink@3900x 26-03-13T17:26:21.177Z:~/data/prgs/rust/vc-template-x1 ((jj/keep/1a79f803025f75fb557a7b6f9d29e3dbee6a1724))
$ vi README.md 
wink@3900x 26-03-13T17:28:08.833Z:~/data/prgs/rust/vc-template-x1 ((jj/keep/1a79f803025f75fb557a7b6f9d29e3dbee6a1724))
$ jj log
@  vnsyoswv wink@saville.com 2026-03-13 10:28:15 main* 3ac24f49
│  feat: Update README.md
◆  vuwzvmwm wink@saville.com 2026-03-13 09:38:22 main@origin 1a79f803
│  feat: Initial commit for the vibe coding main repo
~
wink@3900x 26-03-13T17:28:15.704Z:~/data/prgs/rust/vc-template-x1 ((jj/keep/1a79f803025f75fb557a7b6f9d29e3dbee6a1724))
$ jj git push
Changes to push to origin:
  Move forward bookmark main from 1a79f803025f to 3ac24f49321b
git: Enumerating objects: 5, done.
git: Counting objects: 100% (5/5), done.
git: Delta compression using up to 24 threads
git: Compressing objects: 100% (3/3), done.
git: Writing objects: 100% (3/3), 790 bytes | 790.00 KiB/s, done.
git: Total 3 (delta 2), reused 0 (delta 0), pack-reused 0 (from 0)
remote: Resolving deltas: 100% (2/2), completed with 2 local objects.
Warning: The working-copy commit in workspace 'default' became immutable, so a new commit has been created on top of it.
Working copy  (@) now at: kywoutls c26d415e (empty) (no description set)
Parent commit (@-)      : vnsyoswv 3ac24f49 main | feat: Update README.md
wink@3900x 26-03-13T17:28:33.741Z:~/data/prgs/rust/vc-template-x1 ((main))
```

## Example of modifying an existing commit and "force" push

Tweak a commit and push it using `jj edit` then "force" push:

Minimum steps changing xx but it could be any commit on main
or other bookmark/branch the last step repositions @ so @- is main:

```
jj edit -r xxx --ignore-immutable
<Modify the commit such as, `jj describe or `vi README.md`>
jj git push --bookmark main
jj new main
```

A complete example, the `jj log` commands are to just give
a little more visibility. The thing I'm changing is the conventaional
commit type for of vnsyoswv is "feat" is should be "docs":
```
wink@3900x 26-03-13T17:32:17.819Z:~/data/prgs/rust/vc-template-x1 ((jj/keep/1a79f803025f75fb557a7b6f9d29e3dbee6a1724))
$ jj log -r ::@
@  uxuqmtov wink@saville.com 2026-03-13 10:53:15 d4205bc4
│  (empty) (no description set)
◆  plkoouwq wink@saville.com 2026-03-13 10:50:54 main e76950c0
│  docs: Update README.md with force push example
◆  vnsyoswv wink@saville.com 2026-03-13 10:32:32 525123b1
│  feat: Update README.md
◆  vuwzvmwm wink@saville.com 2026-03-13 09:38:22 1a79f803
│  feat: Initial commit for the vibe coding main repo
◆  zzzzzzzz root() 00000000
wink@3900x 26-03-13T17:57:13.692Z:~/data/prgs/rust/vc-template-x1 ((main))
$ jj edit -r vn --ignore-immutable 
Working copy  (@) now at: vnsyoswv 525123b1 feat: Update README.md
Parent commit (@-)      : vuwzvmwm 1a79f803 feat: Initial commit for the vibe coding main repo
Added 0 files, modified 1 files, removed 0 files
wink@3900x 26-03-13T17:57:27.856Z:~/data/prgs/rust/vc-template-x1 ((jj/keep/1a79f803025f75fb557a7b6f9d29e3dbee6a1724))
Rebased 1 descendant commits
Working copy  (@) now at: vnsyoswv 1b6ed25c docs: Update README.md
Parent commit (@-)      : vuwzvmwm 1a79f803 feat: Initial commit for the vibe coding main repo
wink@3900x 26-03-13T17:58:34.975Z:~/data/prgs/rust/vc-template-x1 ((jj/keep/1a79f803025f75fb557a7b6f9d29e3dbee6a1724))
$ jj log
○  plkoouwq wink@saville.com 2026-03-13 10:58:34 main* bc66029d
│  docs: Update README.md with force push example
@  vnsyoswv wink@saville.com 2026-03-13 10:57:53 1b6ed25c
│  docs: Update README.md
│ ◆  plkoouwq/1 wink@saville.com 2026-03-13 10:50:54 main@origin e76950c0 (hidden)
│ │  docs: Update README.md with force push example
│ ~  (elided revisions)
├─╯
◆  vuwzvmwm wink@saville.com 2026-03-13 09:38:22 1a79f803
│  feat: Initial commit for the vibe coding main repo
~
wink@3900x 26-03-13T18:15:39.052Z:~/data/prgs/rust/vc-template-x1 ((jj/keep/1a79f803025f75fb557a7b6f9d29e3dbee6a1724))
$ jj log -r ::main
○  plkoouwq wink@saville.com 2026-03-13 10:58:34 main* bc66029d
│  docs: Update README.md with force push example
@  vnsyoswv wink@saville.com 2026-03-13 10:57:53 1b6ed25c
│  docs: Update README.md
◆  vuwzvmwm wink@saville.com 2026-03-13 09:38:22 1a79f803
│  feat: Initial commit for the vibe coding main repo
◆  zzzzzzzz root() 00000000
wink@3900x 26-03-13T18:17:20.926Z:~/data/prgs/rust/vc-template-x1 ((jj/keep/1a79f803025f75fb557a7b6f9d29e3dbee6a1724))
$ jj git push --bookmark main
Changes to push to origin:
  Move sideways bookmark main from e76950c0c352 to bc66029d050c
git: Enumerating objects: 8, done.
git: Counting objects: 100% (8/8), done.
git: Delta compression using up to 24 threads
git: Compressing objects: 100% (6/6), done.
git: Writing objects: 100% (6/6), 3.50 KiB | 3.50 MiB/s, done.
git: Total 6 (delta 3), reused 0 (delta 0), pack-reused 0 (from 0)
remote: Resolving deltas: 100% (3/3), completed with 1 local object.
Warning: The working-copy commit in workspace 'default' became immutable, so a new commit has been created on top of it.
Working copy  (@) now at: srxnytso 22165d77 (empty) (no description set)
Parent commit (@-)      : vnsyoswv 1b6ed25c docs: Update README.md
wink@3900x 26-03-13T18:19:07.922Z:~/data/prgs/rust/vc-template-x1 ((jj/keep/1b6ed25cf716ba3686bed15085f0463590a6200c))
$ 
wink@3900x 26-03-13T18:22:21.776Z:~/data/prgs/rust/vc-template-x1 ((jj/keep/1b6ed25cf716ba3686bed15085f0463590a6200c))
$ jj new main
Working copy  (@) now at: vytkmroy 8df04518 (empty) (no description set)
Parent commit (@-)      : plkoouwq bc66029d main | docs: Update README.md with force push example
Added 0 files, modified 1 files, removed 0 files
wink@3900x 26-03-13T18:25:23.243Z:~/data/prgs/rust/vc-template-x1 ((main))
$ jj log -r ::@
@  vytkmroy wink@saville.com 2026-03-13 11:25:23 8df04518
│  (empty) (no description set)
◆  plkoouwq wink@saville.com 2026-03-13 10:58:34 main bc66029d
│  docs: Update README.md with force push example
◆  vnsyoswv wink@saville.com 2026-03-13 10:57:53 1b6ed25c
│  docs: Update README.md
◆  vuwzvmwm wink@saville.com 2026-03-13 09:38:22 1a79f803
│  feat: Initial commit for the vibe coding main repo
◆  zzzzzzzz root() 00000000
wink@3900x 26-03-13T18:25:46.005Z:~/data/prgs/rust/vc-template-x1 ((main))
$
```

## Why `jj log` shows fewer commits than `gitk`

If you're coming from git, jj's log output can be surprising compared to
tools like `gitk --all`.

jj tracks *changes* (identified by change IDs), not individual git commits.
When you rewrite a change (`jj describe`, `jj rebase`, `jj squash`, etc.),
jj creates a new git commit and keeps the old one under `refs/jj/keep/*` as
undo history. `gitk --all` sees all of these obsolete commits; `jj log` only
shows the current version of each change.

## Useful commands

| Command | Description |
|---------|-------------|
| `jj log` | Show recent visible commits (default revset) |
| `jj log -r ::@` | Show **all** ancestors of the working copy |
| `jj log -r 'all()'` | Show all non-hidden commits (needed if you have multiple heads/branches) |
| `jj st | Show the status of the Working and Parent commits |
| `jj st -r <chid> | Status of the commit, <chid> such as `@`, `@-`, `xyz` |
| `jj show | Show the Working commit, -r @ |
| `jj show -r <chid> | Show the commit, <chid> such as `@`, `@-`, `xyz` |
| `jj evolog -r <chid>` | Show the evolution history of a single change |
| `jj op log` | Show operation history (each rewrite operation) |


In a single-branch workflow, `jj log -r ::@` and `jj log -r 'all()'` give
the same result. Use `all()` when you have multiple branches or heads.

## Cross-repo Linking with Git Trailers

Commits in each repo use [git trailers](https://git-scm.com/docs/git-interpret-trailers)
to cross-reference their counterpart in the other repo. The `ochid`
(Other Change ID) trailer contains a workspace-root-relative path
and jj changeID:

```
ochid: /.claude/xvzvruqo   # points to a .claude repo change
ochid: /wtpmottv            # points to an app repo change
```

Paths always start with `/` (the workspace root, i.e. vc-x1).
Each repo has a `.vc-config.toml` that identifies its location
within the workspace, so tools can resolve these paths locally.

For full details see:
- [Git trailer convention](./notes/chores-01.md#git-trailer-convention)
  — [ochid (Other Change ID)](./notes/chores-01.md#ochid-other-change-id)
  — [ChangeID path syntax](./notes/chores-01.md#changeid-path-syntax)
  — [.vc-config.toml](./notes/chores-01.md#vc-configtoml)

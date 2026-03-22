# Vibe Coding Notes

## Use footers to track changeIDs or notes

> **Note:** The original approach used markdown reference-link syntax
> in commit footers. We now use
> [git trailers](https://git-scm.com/docs/git-interpret-trailers)
> (`key: value` format) instead, as they are a standard convention
> and parseable by `git interpret-trailers` and other tools.

Since jj changeIDs are generated at `jj git init` time (not stored in the
repo), changeID references are only resolvable by tools that have access
to the local jj repo (e.g. vc-x1).

## Git trailer convention

We use [git trailers](https://git-scm.com/docs/git-interpret-trailers)
in commit messages for inter-repo cross-references. Trailers appear as
blank-line-separated `key: value` lines at the end of the commit body.

### ChangeID path syntax

All changeID paths are **workspace-root relative** (start with `/`):

- `/` is the workspace root (the app repo, vc-x1)
- `/.claude` is the bot session sub-repo

This means `ochid: /wtpmottv` refers to a change in the app repo,
**not** the .claude repo. The leading `/` anchors to the workspace
root, not the current repo.

### ochid (Other Change ID)

The `ochid` trailer links a commit to its counterpart in another repo
within the workspace. The value is a workspace-root-relative path
followed by the jj changeID:

- `ochid: /changeID` — references the workspace-root repo (vc-x1)
- `ochid: /.claude/changeID` — references the .claude sub-repo

Example commit message:
```
Add jj tips for git users to README

Add jj tips section to README explaining why jj log
shows fewer commits than gitk.

ochid: /.claude/xvzvruqo
```
### BREAKING-CHANGE trailer

During this work we confirmed that `BREAKING CHANGE:` (with space) is the only
space-separated git trailer key allowed per the Conventional Commits spec. We
adopted the hyphenated form `BREAKING-CHANGE:` as it's also valid and avoids
the space ambiguity.

### .vc-config.toml

Each repo contains a `.vc-config.toml` that identifies its location
within the workspace. This avoids repeating the workspace-path in
every commit trailer.

```toml
# In vc-x1 (workspace root):
[workspace]
path = "/"

# In .claude (sub-repo):
[workspace]
path = "/.claude"
```


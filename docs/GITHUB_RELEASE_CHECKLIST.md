# GitHub Release Checklist

This repository snapshot is technically prepared but must not be published
until the following human approvals are complete.

## Required Before Public Push

- [ ] Confirm with LogXY and the supervisor whether the repository may be
  public and select a project-wide code license.
- [ ] Review internship reports for company-confidential information.
- [ ] Confirm redistribution permission for the two small sample images.
- [ ] Decide whether to retain the permitted non-commercial DEIMv2 source
  snapshot or replace it with pinned clone instructions.
- [ ] Review the staged file list before the first commit.
- [ ] Configure the intended GitHub remote explicitly.

## Automated/Local Checks Completed

- Large datasets, annotations, feature caches, checkpoints, raw outputs,
  virtual environments, papers, presentations, credentials, and logs are
  excluded by `.gitignore`.
- No model/data binary matching the prohibited extensions is present in the
  candidate tree.
- JSON and experiment-registry CSV files parse successfully.
- Relative Markdown links resolve.
- The maintained test suite passed: 186 passed, 3 skipped.
- No commit or push has been performed and no remote is configured.

## Suggested First-Commit Review

```bash
git status --short
git add -n .
git diff --no-index /dev/null README.md
```

After approval, stage intentionally and inspect `git diff --cached --stat` plus
`git diff --cached` before committing.

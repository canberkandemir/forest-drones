# GitHub Release Checklist

This repository snapshot was reviewed on 2026-10-24. It is technically
organized for review, but it must not be published until the following human
approvals are complete.

## Required Before Public Push

- [ ] Confirm with LogXY and the supervisor whether the repository may be
  public and select a project-wide code license.
- [ ] Review internship reports for company-confidential information.
- [ ] Confirm redistribution permission for the two small sample images.
- [ ] Decide whether to retain the permitted non-commercial DEIMv2 source
  snapshot or replace it with pinned clone instructions.
- [ ] Review the staged file list before the first commit.
- [ ] Confirm that the existing `origin` remote is the intended destination.
- [ ] Decide whether historical `PROGRESS.md` detail is appropriate for a
  public repository or should remain in a supervisor-only branch/archive.

## Automated/Local Checks Completed

- Large datasets, annotations, feature caches, checkpoints, prediction dumps,
  virtual environments, papers, presentations, credentials, logs, profiler
  traces, experiment-tracker state, archives, and common model binaries are
  excluded by `.gitignore`.
- Lightweight subset manifests under `outputs/subsets/` may be published after
  a human confirms that they contain no restricted metadata or local paths.
- The tracked tree contains no checkpoint/archive binary and no file larger
  than 1 MiB. The largest tracked file is an upstream DEIMv2 notebook.
- Secret-pattern scanning found no likely API key, token, or password.
- Personal absolute macOS paths were replaced in public project documentation;
  `/content/drive/...` remains only in the intentional Colab workflow.
- The experiment-registry CSV parses with 64 unique experiment keys.
- The current publishable candidate is approximately 3.95 MiB across 428
  tracked or non-ignored files; 89 Markdown files have no missing relative
  links.
- Python source files parse statically. No training, inference, or model
  evaluation was run during release preparation.
- No `git add`, commit, push, remote creation, or artifact deletion occurred.

## Suggested First-Commit Review

```bash
git status --short
git add -n .
git diff --check
git diff -- README.md .gitignore docs/GITHUB_RELEASE_CHECKLIST.md
git ls-files -ci --exclude-standard
```

After approval, stage intentionally and inspect `git diff --cached --stat` plus
`git diff --cached` before committing.

## Preserve Without Publishing

- Local datasets and official gated annotations.
- DINOv3, YOLO, DEIMv2, and experiment checkpoints.
- Cached features, raw predictions, training logs, and generated galleries.
- Virtual environments and Hugging Face credentials/cache.
- Interrupted and legacy outputs referenced by the experiment registry.

No repository files were moved or deleted during this cleanup. Legacy root
entry points (`test_dinov3.py`, `test_yolo.py`) and prepared CUDA/DEIMv2
workflows remain because they are documented historical or reproducibility
artifacts, not demonstrated duplicates.

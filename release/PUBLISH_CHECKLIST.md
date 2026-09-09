# GitHub publication checklist

Before pushing the repository, complete these checks on the target branch.

1. Review `git status --short` and confirm that no WSI, feature tensor,
   SQLite database, cache, log, virtual environment or local result directory
   is staged.
2. Mount the old result storage and run
   `scripts/prepare_github_release.py --copy-weights` with the relevant result
   roots. Inspect `release/experiment_artifacts.json` and
   `release/sha256sums.txt`.
3. Install Git LFS and verify the checkpoint files are tracked by LFS:
   `git lfs install` followed by `git lfs ls-files`.
4. Run the review API tests, compile checks and the available real WSI tests.
5. Search the staged payload for local paths, usernames, patient identifiers,
   access tokens and unpublished labels before a public push.
6. Record the exact dataset version, split version, feature version, Python
   environment and checkpoint checksums in the release notes.

The current workspace has no mounted USB2 result root, so the present release
inventory intentionally contains zero copied training checkpoints. This is a
blocker for claiming that the GitHub package contains the historical weights,
but it does not block publishing the code-only package.

# External dependencies

The local workspace contains working copies of upstream repositories for
development. The GitHub repository should fetch or install these dependencies
separately rather than embedding nested `.git` directories.

| Component | Source | Local reference |
| --- | --- | --- |
| CLAM | https://github.com/mahmoodlab/CLAM | `53e2409d4a8189c682c173382964a85f114f923c` |
| Patch-GCN | https://github.com/mahmoodlab/Patch-GCN | `823addaee5b8f4cc2bec3ea8e5e0077b2a5115a4` |
| MIST | https://github.com/Caisner/MIST | vendored under `third_party/mist/`, source revision `229f77c` |

The repository's custom `dsmil/`, `transmil/`, `transmil_official/`,
`patch_gcn/`, scripts and configuration files are included directly. The
upstream CLAM and Patch-GCN working copies, their datasets and their result
checkpoints are intentionally excluded by `.gitignore`.

On a new machine, clone the two upstream projects at the recorded revisions
with `scripts/bootstrap_upstream_dependencies.sh`, then set the corresponding
`CLAM_ROOT` or `PATCH_GCN_ROOT` path in a local environment file if needed. Do
not commit local path overrides or access tokens.

# Five-fold split files

All 14 benchmark configurations must use exactly the same case assignment.
The historical pipeline expects the original filenames below, including the
legacy `flod` spelling:

```text
flod-0-train.csv  flod-0-val.csv  flod-0-test.csv
flod-1-train.csv  flod-1-val.csv  flod-1-test.csv
flod-2-train.csv  flod-2-val.csv  flod-2-test.csv
flod-3-train.csv  flod-3-val.csv  flod-3-test.csv
flod-4-train.csv  flod-4-val.csv  flod-4-test.csv
```

The original split files are stored with the private dataset and are not
currently mounted. Copy them here only for a private repository. For a public
repository, publish anonymized IDs plus class/fold counts, and retain the
private ID mapping outside GitHub.

`fold-4` is the canonical benchmark name for the historical fifth fold that
was originally stored under the `fold5_hp+yx` result root.

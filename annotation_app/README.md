# HALO Annotation

The Streamlit platform retains its original login, administrator and Audit/Eval
pages. [`data/`](data/) also preserves the exported **8,000 Eval annotations** and
**800 Audit reviews** used in the research; those CSVs are independent of the
platform's current sample files and database.

This release clears the old database accounts, assignments, saved reviews and
assignment repair records once, using the permanent reset marker
`user-requested-empty-platform-2026-10-14`. Later restarts, cache refreshes and
code updates do not repeat this reset. The administrator configured through
Streamlit Secrets remains available to create new accounts.

Sample lists start empty. Supply `audit-items.json` and/or `eval-items.json`
under `data/` (gzip files with `.json.gz` are also supported), or set
`ANNOTATION_DATA_DIR` to an external directory. Each file accepts the original
item dictionaries as a JSON list or an object containing an `items` list.
The exported annotation CSVs are never loaded as samples.

Database and administrator configuration still use
[`secrets.example.toml`](secrets.example.toml). Keep real credentials outside Git.
Run from the repository root with `streamlit run annotation_app/app.py`.

Changed code and sample-file signatures invalidate affected caches; widget
reruns keep the shared resources. Existing sessions discard old login state on
loading this entrypoint version. A cloud process must load the new code for the
restored pages and one-time reset to take effect.

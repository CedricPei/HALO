# HALO Annotation

Human annotation is complete. [`data/`](data/) preserves **8,000 Eval
annotations** and **800 Audit reviews**. Platform screenshots appear in the
[project README](../README.md#human-annotation).

The platform implementation is retained as a reference. `app.py` now displays
a completion page without login, task assignment or annotation controls. It
does not connect to the database; existing accounts, assignments and saved
judgments remain untouched.

The two former human-evaluation sample catalogs have been removed. The exported
annotation CSVs retain their original contents.

On its first rerun after this update, each existing Streamlit session clears the
old platform caches and login state once for the retirement version. Later
reruns retain that version marker. A cloud process must load the updated
entrypoint before the completion page takes effect; restart it if deployment
does not trigger a rerun.

# HALO Annotation

This directory contains the Streamlit application and [`results/`](results/),
with 8,000 exported Eval annotations and 800 Audit reviews.

The app reads the two original frozen response batches from
[`eval_results/`](../eval_results/). Accounts, credentials, task assignments and
live judgments are stored in the database configured by `DATABASE_URL`. Account
setup is managed through the administrator page; restarting the app does not
recreate the old participant accounts or replay their initialization migrations.
The result CSVs are snapshots of human judgments.

Configure the app through environment variables or Streamlit Secrets using
[`secrets.example.toml`](secrets.example.toml). Keep real database credentials,
administrator passwords and participant handouts outside Git.

Run from the repository root:

```bash
streamlit run annotation_app/app.py
```

Frozen response IDs, their display order and result CSV contents remain
unchanged. The app versions its release cache by loader source and file signature,
so a changed loader reloads on the next Streamlit rerun without reloading the
responses on every click.

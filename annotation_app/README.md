# Proactive Agent Benchmark Annotation Interface

This is the cloud-ready version of the existing local annotation pilot. It
does not modify or replace `interfaces/`, the five `pilot/reviewer*` folders,
or their local SQLite workflow.

## What changes in the App version

- Audit and Eval are available from one Streamlit URL.
- Annotators sign in with a participant ID and random access code; emails are
  not required.
- Supabase PostgreSQL stores accounts, assignments, progress, and annotations.
- Benchmark JSON, model outputs, and instance state remain read-only files.
- Every save appends an immutable revision instead of overwriting the previous
  annotation.
- Supabase Row Level Security is enabled with no public policies; only the
  server-side App connection can read or write annotation tables.
- An administrator can create one account, generate a balanced batch, recreate
  the current five pilot assignment packs, monitor progress, disable accounts,
  reset codes, and export the latest annotations.

## 1. Preview locally without Supabase

Install the App dependencies:

```bash
python -m pip install -r annotation_app/requirements.txt
```

Generate two local secrets:

```bash
python -m annotation_app.generate_secrets
```

Set the printed values as environment variables, then run:

```bash
export ADMIN_PASSWORD='the generated admin password'
export AUTH_PEPPER='the generated auth pepper'
streamlit run annotation_app/app.py
```

With no `DATABASE_URL`, this App writes only to the ignored local file
`annotation_app/local/annotations.sqlite`. The existing local annotation
database is not touched.

## 2. Connect the existing Supabase project

In the Supabase project dashboard:

1. Click **Connect**.
2. Select the pooled/transaction connection string. This normally uses port
   `6543` and works from Streamlit Community Cloud.
3. Replace the password placeholder with the database password chosen when the
   project was created.
4. Keep the completed connection string private.

The App creates the tables in `schema.sql` automatically on its first database
connection. No manual table creation is required.

For a local cloud-database test, set all three secrets:

```bash
export DATABASE_URL='the Supabase pooled connection string'
export ADMIN_PASSWORD='a long private admin password'
export AUTH_PEPPER='a second long random private value'
streamlit run annotation_app/app.py
```

## 3. Deploy one private GitHub repository as one App

1. Push the finished source and selected benchmark data to the private GitHub
   repository.
2. Sign in at <https://share.streamlit.io/> with GitHub.
3. Create one App from the repository.
4. Choose `annotation_app/app.py` as the entrypoint.
5. In **Advanced settings > Secrets**, paste values following
   `annotation_app/secrets.example.toml`.
6. Deploy and sign in with `ADMIN_USERNAME` and `ADMIN_PASSWORD`.

Do not add `secrets.toml`, the Supabase password, or a real `DATABASE_URL` to
GitHub.

## 4. Create annotator access

In the Administration page, choose **Audit** or **Eval** and enter the number
of annotators. The App creates sequential participant IDs and random access
codes, then distributes that task type evenly across the batch. Each account
receives only one task type, and all Event Prefixes belonging to one Eval
Scenario remain assigned together. Audit Scenarios receive two independent
reviews whenever the batch contains at least two annotators.

Download the generated account CSV immediately. Only secure hashes of access
codes are stored, so the original codes cannot be recovered later; an admin can
generate a replacement code when needed.

## 5. Production review safeguards

New Audit and Eval items start without a selected verdict, and the reviewer
cannot save or advance until selecting one. Annotator task pages also block
ordinary text selection, copy/cut actions, dragging, and the context menu.
These browser-side controls discourage casual transfer to external tools but
cannot prevent screenshots, photographs, or deliberate extraction through
developer tools.

## Import completed local annotations

First create the corresponding participant and assignments in the App. Then,
with the target `DATABASE_URL` configured, run:

```bash
python -m annotation_app.import_local_reviews \
  /absolute/path/to/annotations/reviews.sqlite \
  --user-id reviewer1
```

The importer skips unassigned items and exact duplicates. It does not modify
the source SQLite file.

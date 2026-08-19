# Instance Creator App API

Represent every call as a JSON command with lowercase `app` and `function`
values. `parameter` is an object keyed by the named arguments shown below;
`database.execute` may instead receive its SQL string directly.

## database

- `execute(sql, parameters=())`: Execute one SQLite statement. When `parameter`
  is an object, use `sql` and optional `parameters`; `parameters` may be a list
  or object.

## files

- `create_file(name, content="")`: Create one uniquely named plain `.txt` File.

## wallet

- `setup_wallet(currency, balance=0, account_type="personal", institution=None)`:
  Initialize a Wallet.

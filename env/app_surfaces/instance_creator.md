# Instance Creator App API

## Database

- `execute(sql, parameters=())`: Execute one SQLite statement. `parameters` may
  be a list or object.

## Files

- `create_file(name, content="")`: Create one uniquely named plain `.txt` File.

## Wallet

- `setup_wallet(currency, balance=0, account_type="personal", institution=None)`:
  Initialize a Wallet.

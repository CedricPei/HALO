# Scenario Creator App API

Timestamps use ISO 8601 with a timezone whenever possible. Natural keys must
remain unique. Operations on a missing object raise an error.

## Database

- `execute(sql, parameters=())`: Execute one SQLite statement. Row-producing
  statements return `columns` and `rows`; other statements return
  `affected_rows`. Use parameterized SQL when values are variable.

## Files

- `create_file(name, content="")`: Create a uniquely named plain `.txt` File.
- `append_file(name, content)`: Append content without replacing existing text.
- `replace_file(name, content)`: Replace the entire content of an existing File.
- `delete_file(name)`: Delete an existing File.

## Email

An Email is identified by `sender + subject + timestamp`. A draft belongs to the
Instance user and is identified by its creation `timestamp`.

- `create_email(sender, recipients, subject, body, timestamp, cc=None)`: Create
  a sent or received Email record.
- `delete_email(sender, subject, timestamp)`: Delete an Email record.
- `flag_email(sender, subject, timestamp, comment)`: Flag an existing Email and
  store a non-empty explanation of why it needs attention.
- `create_draft(recipients, subject, body, timestamp, cc=None)`: Create an
  outgoing Email draft as the Instance user without sending it.
- `update_draft(timestamp, **changes)`: Update the Instance user's draft.
  Editable fields are `recipients`, `subject`, `body`, and `cc`.

## Calendar

An event is identified by `title + start_time`; its `end_time` must be later than
its start. Event drafts are separate from active events. A reminder is identified
by `title + remind_time`, and `remind_time` cannot be later than `due_time`.

- `create_event(title, start_time, end_time, description=None, location=None, attendees=None, organizer=None, status=None)`:
  Create an active event. Common statuses are `confirmed`, `tentative`, and
  `cancelled`.
- `update_event(title, start_time, **changes)`: Update an active event selected
  by its current natural key.
- `delete_event(title, start_time)`: Delete an active event.
- `create_event_draft(title, start_time, end_time, description=None, location=None, attendees=None, organizer=None, status=None)`:
  Create an event draft that does not enter the active calendar.
- `update_event_draft(title, start_time, **changes)`: Update an event draft.
- `delete_event_draft(title, start_time)`: Delete an event draft.
- `create_reminder(title, remind_time, content, due_time=None, completed=False)`:
  Create a reminder with non-empty user-visible content.
- `update_reminder(title, remind_time, **changes)`: Update a reminder; use
  `completed=True` to mark it complete.
- `delete_reminder(title, remind_time)`: Delete a reminder.

## Messaging

A conversation is identified by its complete participant set, regardless of
order. A conversation must exist before creating a message or draft in it.

- `create_conversation(participants, title=None)`: Create a conversation, or
  return the existing conversation with the same participant set.
- `create_message(sender, recipients, content, timestamp)`: Create a message in
  an existing conversation.
- `update_message(sender, recipients, timestamp, content)`: Replace an existing
  message's content.
- `pin_conversation(participants, comment)`: Pin an existing conversation and
  store a non-empty explanation of why it needs attention.
- `create_draft(recipients, content, timestamp)`: Create a message draft as the
  Instance user in an existing conversation.
- `update_draft(recipients, timestamp, content)`: Replace the Instance user's
  selected message draft content.

## Contacts

Contact names are unique natural keys.

- `create_contact(name, aliases=None, emails=None, phone_numbers=None, organization=None, relationship=None, notes=None)`:
  Create a contact.
- `update_contact(name, **changes)`: Update a contact selected by its current
  name.
- `flag_contact(name, comment)`: Flag an existing contact and store a non-empty
  explanation of why the user should follow up.
- `delete_contact(name)`: Delete a contact.

## Wallet

Amounts must be positive. `completed` payments reduce balance; completed
deposits and refunds increase it. Transactions are selected by their generated
`transaction_id` and cannot be deleted.

- `update_wallet(account_type=None, institution=None)`: Update Wallet metadata;
  it cannot change balance or currency.
- `add_funds(amount, timestamp, source="manual", status="completed", reference=None, description=None)`:
  Record incoming funds.
- `pay(merchant, amount, timestamp, status="completed", category=None, reference=None, description=None)`:
  Record a payment.
- `update_transaction_status(transaction_id, status)`: Change a payment or
  deposit among `pending`, `completed`, and `failed`.
- `refund_transaction(transaction_id, timestamp, description=None)`: Refund a
  completed, unrefunded payment.
- `flag_transaction(transaction_id, comment)`: Flag an existing transaction and
  store a non-empty explanation without changing its status or balance effect.
- `create_draft(merchant, amount, timestamp, category=None, reference=None, description=None)`:
  Create a payment draft without changing balance.
- `update_draft(timestamp, **changes)`: Update a payment draft's `merchant`,
  `amount`, `category`, `reference`, or `description`.

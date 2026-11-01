# Evaluated Agent App API

This is the complete command surface available to the evaluated Agent. A
command that is not listed here is unavailable.

## Inspect commands

These commands retrieve information without changing App state. Database and
Files are read-only.

### Database

- `database.get_schema()`
- `database.execute(sql, parameters=())`: Only `SELECT`, `WITH`, and `PRAGMA` statements
  are allowed.

### Files

- `files.get_file(name)`
- `files.list_files()`
- `files.search_files(query)`

### Email

- `email.get_email(sender, subject, timestamp)`
- `email.list_emails(sender=None, recipient=None, start_time=None, end_time=None, flagged=None)`
- `email.search_emails(query)`
- `email.list_drafts(query=None)`

### Calendar

- `calendar.get_event(title, start_time)`
- `calendar.list_events(query=None, start_time=None, end_time=None, status=None, attendee=None)`
- `calendar.find_conflicts(start_time, end_time, include_cancelled=False)`
- `calendar.get_event_draft(title, start_time)`
- `calendar.list_event_drafts(query=None, start_time=None, end_time=None, attendee=None)`
- `calendar.get_reminder(title, remind_time)`
- `calendar.list_reminders(query=None, start_time=None, end_time=None, completed=None)`

### Messaging

- `messaging.get_conversation(participants)`
- `messaging.list_conversations(participant=None, query=None, pinned=None)`
- `messaging.get_message(sender, recipients, timestamp)`
- `messaging.list_messages(participants=None, sender=None, start_time=None, end_time=None)`
- `messaging.search_messages(query, participants=None, sender=None)`
- `messaging.list_drafts(participants=None, query=None)`

### Contacts

- `contacts.get_contact(name)`
- `contacts.list_contacts(flagged=None)`
- `contacts.search_contacts(query)`

### Wallet

- `wallet.get_wallet()`
- `wallet.get_transaction(transaction_id)`
- `wallet.list_transactions(query=None, transaction_type=None, status=None, start_time=None, end_time=None, min_amount=None, max_amount=None, flagged=None)`
- `wallet.list_drafts(query=None, min_amount=None, max_amount=None)`

## Action commands

### Assistant

- `assistant.notify_user(message)`: Present a non-empty message to the Instance
  user at the current checkpoint (L1 attention). The recipient and timestamp
  are supplied automatically. This is a one-way notification: it provides no
  user reply or approval, contacts no third party, and performs none of the
  operations proposed in its text. Ground the message in available evidence
  and provide useful information or a concrete next step; generic advice alone
  does not establish that the current need has been addressed.

### Email

- `email.flag_email(sender, subject, timestamp, comment)`: Flag an existing
  Email with a non-empty explanation of why it needs attention.
- `email.create_draft(recipients, subject, body, cc=None)`: Create an outgoing
  Email draft as the Instance user without sending it. The creation time is
  supplied automatically.
- `email.send_email(recipients, subject, body, cc=None)`: Send as the Instance
  user; the sender and sending time are supplied automatically.

### Calendar

- `calendar.create_reminder(title, remind_time, content, due_time=None)`: Create
  a reminder with non-empty user-visible content describing what to remember.
- `calendar.create_event_draft(title, start_time, end_time, description=None, location=None, attendees=None, organizer=None, status=None)`:
  Create an event draft outside the active calendar and conflict detection.
- `calendar.create_event(title, start_time, end_time, description=None, location=None, attendees=None, organizer=None, status=None)`:
  Create an active Calendar event.
- `calendar.update_event(title, start_time, **changes)`: Update an active event.
- `calendar.delete_event(title, start_time)`: Delete an active event.

### Messaging

- `messaging.pin_conversation(participants, comment)`: Pin an existing
  conversation with a non-empty explanation of why it needs attention.
- `messaging.create_draft(recipients, content)`: Create a message draft as the
  Instance user in an existing conversation without sending it. The creation
  time is supplied automatically.
- `messaging.send_message(recipients, content)`: Send as the Instance user in an
  existing conversation; the sender and sending time are supplied automatically.

### Contacts

- `contacts.flag_contact(name, comment)`: Flag an existing contact with a
  non-empty explanation of why the user should follow up.
- `contacts.create_contact(name, aliases=None, emails=None, phone_numbers=None, organization=None, relationship=None, notes=None)`:
  Create an internal contact record.
- `contacts.update_contact(name, **changes)`: Update an internal contact record.

### Wallet

- `wallet.flag_transaction(transaction_id, comment)`: Flag an existing
  transaction with a non-empty explanation, without changing status or balance.
- `wallet.create_draft(merchant, amount, category=None, reference=None, description=None)`:
  Create a payment draft without a transaction or balance change. The creation
  time is supplied automatically.
- `wallet.pay(merchant, amount, status="completed", category=None, reference=None, description=None)`:
  Create a payment with an automatically supplied transaction time; a completed
  payment reduces balance.
- `wallet.refund_transaction(transaction_id, description=None)`: Refund a
  completed, unrefunded payment at the automatically supplied transaction time
  and restore its amount.

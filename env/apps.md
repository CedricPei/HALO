# Apps

## General conventions

- JSON-backed apps save immediately after every create, update, or delete operation.
- Objects are selected with meaningful fields such as a contact name, filename, or a title and timestamp. These natural keys are kept unique so that lookups are unambiguous.
- Timestamps use ISO 8601 strings. Include a timezone whenever possible, for example `2026-07-25T14:30:00+08:00` or `2026-07-25T06:30:00Z`.
- Time-range filters are inclusive unless an app explicitly describes interval overlap.
- Text search is a case-insensitive substring search, not a regular-expression search.
- A missing object raises `KeyError`. Invalid operations, such as spending more than the available balance, raise `ValueError`.

## Email

Email stores sent and received email records and outgoing drafts.

### Email fields

- `sender`: Address or name that sent the email.
- `recipients`: List of primary recipient addresses or names.
- `subject`: Email subject.
- `body`: Full email body.
- `timestamp`: ISO 8601 email time.
- `cc`: Optional list of copied contacts.

Drafts use the same fields. For a draft, `timestamp` is its creation time.

An email uses `sender + subject + timestamp` as its natural key. A draft uses `sender + timestamp`, allowing its subject to be edited.

### Email functions

- `create_email(sender, recipients, subject, body, timestamp, cc=None)`: Create and return an email with explicit sender and recipients.
- `get_email(sender, subject, timestamp)`: Return an email by its natural key.
- `list_emails(sender=None, recipient=None, start_time=None, end_time=None)`: List emails with optional sender, recipient-membership, and inclusive time filters.
- `search_emails(query)`: Search sender, recipients, copied recipients, subject, and body.
- `delete_email(sender, subject, timestamp)`: Delete and return an email by its natural key.
- `create_draft(sender, recipients, subject, body, timestamp, cc=None)`: Compose and save an outgoing email draft without sending it.
- `list_drafts(query=None)`: List drafts, optionally searching sender, recipients, copied recipients, subject, and body.
- `update_draft(sender, timestamp, **changes)`: Update a draft by its natural key. `recipients`, `subject`, `body`, and `cc` may be changed; `sender` and the creation `timestamp` remain fixed.

## Calendar

Calendar stores events and reminders.

### Event fields

- `title`: Event title.
- `start_time`: ISO 8601 event start time.
- `end_time`: ISO 8601 event end time; it must be later than `start_time`.
- `description`: Optional details.
- `location`: Optional physical or virtual location.
- `attendees`: Optional list of attendee names or addresses.
- `organizer`: Optional organizer name or address. The organizer does not have to be the user who owns the App.
- `status`: Optional status label. Common values are `confirmed`, `tentative`, and `cancelled`. The value `cancelled` is special because cancelled events are excluded from conflict detection by default.

The event natural key is `title + start_time`.

### Event functions

- `create_event(title, start_time, end_time, description=None, location=None, attendees=None, organizer=None, status=None)`: Create and return an event after validating its interval.
- `get_event(title, start_time)`: Return an event by its natural key.
- `list_events(query=None, start_time=None, end_time=None, status=None, attendee=None)`: Search and filter events. A time range matches any event that overlaps the range. `query` searches title, description, location, organizer, and attendees.
- `update_event(title, start_time, **changes)`: Find an event by its current natural key, update it, and validate the resulting interval.
- `delete_event(title, start_time)`: Delete and return an event by its natural key.
- `find_conflicts(start_time, end_time, include_cancelled=False)`: Return events that overlap a proposed interval. Set `include_cancelled=True` to include cancelled events.

### Reminder fields

- `title`: Reminder title.
- `remind_time`: ISO 8601 time when the reminder should trigger.
- `completed`: Boolean indicating whether the task is finished.
- `description`: Optional details.
- `due_time`: Optional ISO 8601 deadline for the underlying task. This differs from `remind_time`, which controls when the reminder appears. The reminder time cannot be later than the due time.

The reminder natural key is `title + remind_time`.

### Reminder functions

- `create_reminder(title, remind_time, completed=False, description=None, due_time=None)`: Create and return a reminder.
- `get_reminder(title, remind_time)`: Return a reminder by its natural key.
- `list_reminders(query=None, start_time=None, end_time=None, completed=None)`: Search title and description and optionally filter by inclusive reminder-time range and completion state.
- `update_reminder(title, remind_time, **changes)`: Find a reminder by its current natural key and update it. Use `completed=True` to mark it complete.
- `delete_reminder(title, remind_time)`: Delete and return a reminder.

## Messaging

Messaging stores one-to-one and group conversations, messages, and outgoing drafts.

### Conversation fields

- `participants`: List of everyone in the conversation. Participant order does not affect conversation identity.
- `title`: Optional conversation title.

A conversation is uniquely determined by its complete participant set.

### Message fields

- `sender`: Person who sent the message.
- `recipients`: List of people who received the message.
- `content`: Message text.
- `timestamp`: ISO 8601 send time.

A message is uniquely selected by `sender + recipients + timestamp`. Recipient order does not affect lookup.

Message drafts use the same fields and natural key. A draft belongs to an existing conversation but has not been sent.

### Messaging functions

- `create_conversation(participants, title=None)`: Create a conversation for a participant list. If the same participant set already exists in another order, return the existing conversation.
- `get_conversation(participants)`: Return the conversation with exactly this participant set.
- `list_conversations(participant=None, query=None)`: List conversations with optional filters. `participant` requires one person to be present; `query` searches participant names and the title.
- `create_message(sender, recipients, content, timestamp)`: Create a message. A conversation containing exactly the sender and recipients must already exist.
- `get_message(sender, recipients, timestamp)`: Return one message by its natural key.
- `list_messages(participants=None, sender=None, start_time=None, end_time=None)`: Return messages sorted by timestamp. `participants` selects an exact conversation regardless of participant order; `sender` filters by the person who sent the message.
- `search_messages(query, participants=None, sender=None)`: Search sender, recipients, and content with optional conversation and sender filters.
- `update_message(sender, recipients, timestamp, content)`: Replace the content of a selected message. Its sender, recipients, and timestamp remain unchanged.
- `create_draft(sender, recipients, content, timestamp)`: Compose a message draft. A conversation containing exactly the sender and recipients must already exist.
- `list_drafts(participants=None, query=None)`: List message drafts with optional exact-conversation and text filters.
- `update_draft(sender, recipients, timestamp, content)`: Replace a selected draft's content without sending it.

## Contacts

Contacts stores address-book records.

### Contact fields

- `name`: Unique contact name and the natural lookup key.
- `aliases`: Optional list of alternate names.
- `emails`: Optional list of email addresses.
- `phone_numbers`: Optional list of phone numbers.
- `organization`: Optional employer or organization.
- `relationship`: Optional relationship description such as `friend`, `coworker`, or `family`.
- `notes`: Optional free-text notes.

### Contact functions

- `create_contact(name, aliases=None, emails=None, phone_numbers=None, organization=None, relationship=None, notes=None)`: Create and return a contact. Names must be unique.
- `get_contact(name)`: Return the contact with an exact name match.
- `list_contacts()`: Return all contacts.
- `search_contacts(query)`: Search every user-visible contact field.
- `update_contact(name, **changes)`: Find a contact by its current name and update it. A changed name must remain unique.
- `delete_contact(name)`: Delete and return a contact by exact name.

## Wallet

Wallet stores one personal account and its transaction history.

### Wallet fields

- `currency`: Currency used by the wallet and all of its transactions, such as `CNY` or `USD`. It is fixed after setup.
- `balance`: Current available balance.
- `account_type`: Account label, such as `personal`, `checking`, or `cash`.
- `institution`: Optional bank or wallet provider.

### Transaction fields

- `transaction_id`: Automatically generated ID used by Wallet transaction lookup, status update, and refund functions.
- `transaction_type`: One of `payment`, `deposit`, or `refund`.
- `amount`: Positive transaction amount. Direction is represented by `transaction_type`, not by a negative number.
- `currency`: Copied from the wallet currency.
- `timestamp`: ISO 8601 transaction time.
- `status`: Transaction lifecycle state.
- `merchant`: Merchant name for payments and refunds.
- `source`: Source label for deposits, such as `salary`, `bank transfer`, or `manual`.
- `category`: Optional payment category.
- `reference`: Optional external reference. Refund transactions use the original payment ID here.
- `description`: Optional free-text note.

Accepted status values are:

- `pending`: Recorded but not finalized; it has no balance effect.
- `completed`: Finalized. A payment decreases balance, while a deposit or refund increases balance.
- `failed`: Unsuccessful; it has no balance effect.
- `refunded`: Final state of an original payment after `refund_transaction`. The original debit remains in history and a separate completed refund transaction credits the same amount.

`pay` and `add_funds` accept `pending`, `completed`, or `failed`. `refunded` is assigned only by `refund_transaction`.

Payment drafts use `merchant`, `amount`, `currency`, `timestamp`, and the optional payment fields. A draft is not a transaction, has no status, and does not affect the wallet balance. Its creation `timestamp` is its natural key.

### Wallet functions

- `setup_wallet(currency, balance=0, account_type="personal", institution=None)`: Initialize the single wallet. It can be called only once for a state file, and the starting balance cannot be negative.
- `get_wallet()`: Return wallet configuration, current balance, and currency without the transaction rows.
- `update_wallet(account_type=None, institution=None)`: Update the account label or institution. Balance is changed only by money operations, and currency remains fixed so transaction history stays consistent.
- `get_transaction(transaction_id)`: Return one transaction by its generated transaction ID.
- `add_funds(amount, timestamp, source="manual", status="completed", reference=None, description=None)`: Record incoming funds. Completed deposits increase balance immediately.
- `pay(merchant, amount, timestamp, status="completed", category=None, reference=None, description=None)`: Record a payment. Completed payments decrease balance immediately and fail when funds are insufficient.
- `create_draft(merchant, amount, timestamp, category=None, reference=None, description=None)`: Prepare a payment draft without creating a transaction or changing the balance.
- `list_drafts(query=None, min_amount=None, max_amount=None)`: List payment drafts with optional text and amount filters.
- `update_draft(timestamp, **changes)`: Update a payment draft's merchant, amount, category, reference, or description. Currency and creation time remain fixed.
- `list_transactions(query=None, transaction_type=None, status=None, start_time=None, end_time=None, min_amount=None, max_amount=None)`: Search and filter transactions, then return them in timestamp order. `query` searches merchant, source, category, reference, and description. `transaction_type` accepts `payment`, `deposit`, or `refund`.
- `update_transaction_status(transaction_id, status)`: Change a payment or deposit between `pending`, `completed`, and `failed`, adjusting balance when its finalized state changes. Refund records and already-refunded payments are final.
- `refund_transaction(transaction_id, timestamp, description=None)`: Refund a completed payment, mark it as `refunded`, create a separate completed refund record, and restore the amount to the balance.

Transactions cannot be deleted because the wallet preserves financial history.

## Database

Database provides access to a SQLite database.

### Database functions

- `get_schema()`: Return all user-created tables. Each column includes its `name`, SQLite `type`, `not_null` flag, `default_value`, and `primary_key` flag.
- `execute(sql, parameters=())`: Execute one SQL statement. Statements returning rows produce `{"columns": [...], "rows": [...]}`. Other statements produce `{"affected_rows": number}`. SQLite failures produce `{"error": message}`. Parameterized SQL may use a tuple, list, or mapping supported by SQLite.

SQLite foreign-key enforcement is enabled for every Database connection. When commands are run through `CommandParser`, an SQL error is raised explicitly instead of being treated as a successful result.

There is no public `close` function.

## Files

Files manages `.txt` documents directly inside its storage directory. It does not create a catalog or metadata file.

### File fields

- `name`: Unique plain filename ending in `.txt`. Directory components are not allowed.
- `content`: Full UTF-8 text stored directly in the TXT file.

The exact `name` is the natural key for file operations.

### File functions

- `create_file(name)`: Create and return an empty TXT file. Names must be unique.
- `get_file(name)`: Return the name and full content of an exact file.
- `list_files()`: Return the name of every managed file.
- `search_files(query)`: Perform a case-insensitive substring search over filename and full content.
- `append_file(name, content)`: Add content to the end of an existing file without changing its current content.
- `replace_file(name, content)`: Replace all existing content in a file with the supplied content.
- `delete_file(name)`: Delete the TXT file and return the deleted data.

## Web

`Web(api_key=None, timeout=30)` performs live Tavily searches and stores no local state. When `api_key` is omitted, the app reads `TAVILY_API_KEY` from the environment. `timeout` is the maximum number of seconds to wait for the HTTP response.

### Web result fields

- `title`: Result page title.
- `url`: Result URL.
- `content`: Raw page content when Tavily provides it, otherwise the search snippet.
- `score`: Tavily relevance score; larger values indicate a more relevant result.

### Web functions

- `search_web(query, max_results=5)`: Run a live search and return a list of result dictionaries. If the API key is missing or the request fails, return `{"error": message}`.
- `open_url(url)`: Open a URL without using the Tavily API. Return the final URL after redirects, HTTP status, and decoded page content as `{"url": ..., "status": ..., "content": ...}`. Request failures return `{"error": message}`.

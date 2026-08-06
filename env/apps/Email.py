"""Email and draft storage, filtering, and search."""

from ._store import JsonStore, in_time_range, keyword_match, next_id, public


class Email:
    """Store sent and received emails and outgoing drafts in a JSON file."""

    def __init__(self, state_path):
        self.store = JsonStore(state_path, {"emails": [], "drafts": []})
        self.emails = self.store.data.setdefault("emails", [])
        self.drafts = self.store.data.setdefault("drafts", [])

    def _email(self, sender, subject, timestamp):
        """Find an email by sender, subject, and timestamp."""
        for email in self.emails:
            if (
                email["sender"] == sender
                and email["subject"] == subject
                and email["timestamp"] == timestamp
            ):
                return email
        raise KeyError(f"Email not found: {sender}, {subject}, {timestamp}")

    def _draft(self, sender, timestamp):
        """Find a draft by sender and creation time."""
        for draft in self.drafts:
            if (
                draft["sender"] == sender
                and draft["timestamp"] == timestamp
            ):
                return draft
        raise KeyError(f"Draft not found: {sender}, {timestamp}")

    def create_email(
        self, sender, recipients, subject, body, timestamp, cc=None
    ):
        """Create and save an email with explicit sender and recipients."""
        if any(
            item["sender"] == sender
            and item["subject"] == subject
            and item["timestamp"] == timestamp
            for item in self.emails
        ):
            raise ValueError("An email with the same sender, subject, and timestamp exists")

        email = {
            "email_id": next_id(self.emails, "email_id", "email"),
            "sender": sender,
            "recipients": recipients,
            "subject": subject,
            "body": body,
            "timestamp": timestamp,
        }
        if cc is not None:
            email["cc"] = cc
        self.emails.append(email)
        self.store.save()
        return public(email)

    def get_email(self, sender, subject, timestamp):
        """Return an email selected by sender, subject, and timestamp."""
        return public(self._email(sender, subject, timestamp))

    def list_emails(
        self, sender=None, recipient=None, start_time=None, end_time=None
    ):
        """List emails with optional sender, recipient, and time filters."""
        return public(
            [
                email
                for email in self.emails
                if (sender is None or email["sender"] == sender)
                and (recipient is None or recipient in email["recipients"])
                and in_time_range(email["timestamp"], start_time, end_time)
            ]
        )

    def search_emails(self, query):
        """Search sender, recipients, copied recipients, subject, and body."""
        return public(
            [
                email
                for email in self.emails
                if keyword_match(
                    query,
                    email["sender"],
                    email["recipients"],
                    email.get("cc", []),
                    email["subject"],
                    email["body"],
                )
            ]
        )

    def delete_email(self, sender, subject, timestamp):
        """Delete an email selected by sender, subject, and timestamp."""
        email = self._email(sender, subject, timestamp)
        self.emails.remove(email)
        self.store.save()
        return public(email)

    def create_draft(
        self, sender, recipients, subject, body, timestamp, cc=None
    ):
        """Compose and save an outgoing email draft."""
        if any(
            item["sender"] == sender
            and item["timestamp"] == timestamp
            for item in self.drafts
        ):
            raise ValueError("A draft with the same sender and timestamp exists")

        draft = {
            "draft_id": next_id(self.drafts, "draft_id", "draft"),
            "sender": sender,
            "recipients": recipients,
            "subject": subject,
            "body": body,
            "timestamp": timestamp,
        }
        if cc is not None:
            draft["cc"] = cc
        self.drafts.append(draft)
        self.store.save()
        return public(draft)

    def list_drafts(self, query=None):
        """List drafts, optionally filtering by a text query."""
        return public(
            [
                draft
                for draft in self.drafts
                if query is None
                or keyword_match(
                    query,
                    draft["sender"],
                    draft["recipients"],
                    draft.get("cc", []),
                    draft["subject"],
                    draft["body"],
                )
            ]
        )

    def update_draft(self, sender, timestamp, **changes):
        """Update a draft selected by sender and creation time."""
        draft = self._draft(sender, timestamp)
        changes.pop("draft_id", None)
        changes.pop("sender", None)
        changes.pop("timestamp", None)
        draft.update(changes)
        self.store.save()
        return public(draft)

"""Group conversations, messages, and message drafts."""

from ._store import (
    JsonStore,
    in_time_range,
    keyword_match,
    next_id,
    parse_time,
    public,
    required_text,
)


class Messaging:
    """Store conversations, messages, and outgoing message drafts."""

    def __init__(self, state_path, owner=None):
        self.store = JsonStore(
            state_path, {"conversations": [], "messages": [], "drafts": []}
        )
        self.conversations = self.store.data.setdefault("conversations", [])
        self.messages = self.store.data.setdefault("messages", [])
        self.drafts = self.store.data.setdefault("drafts", [])
        self.owner = owner

    @staticmethod
    def _same_people(left, right):
        """Compare participant lists without depending on their order."""
        return set(left) == set(right)

    def _conversation(self, participants):
        for conversation in self.conversations:
            if self._same_people(conversation["participants"], participants):
                return conversation
        raise KeyError(f"Conversation not found for: {participants}")

    def _message(self, sender, recipients, timestamp):
        """Find a message by its visible sender, recipients, and timestamp."""
        for message in self.messages:
            if (
                message["sender"] == sender
                and self._same_people(message["recipients"], recipients)
                and message["timestamp"] == timestamp
            ):
                return message
        raise KeyError(f"Message not found: {sender}, {recipients}, {timestamp}")

    def _draft(self, sender, recipients, timestamp):
        """Find a draft by its visible natural key."""
        for draft in self.drafts:
            if (
                draft["sender"] == sender
                and self._same_people(draft["recipients"], recipients)
                and draft["timestamp"] == timestamp
            ):
                return draft
        raise KeyError(f"Draft not found: {sender}, {recipients}, {timestamp}")

    def create_conversation(self, participants, title=None):
        """Create a conversation or return one with the same participants."""
        participants = list(dict.fromkeys(participants))
        for conversation in self.conversations:
            if self._same_people(conversation["participants"], participants):
                return public(conversation)

        conversation = {
            "conversation_id": next_id(
                self.conversations, "conversation_id", "conversation"
            ),
            "participants": participants,
            "pinned": False,
        }
        if title is not None:
            conversation["title"] = title
        self.conversations.append(conversation)
        self.store.save()
        return public(conversation)

    def get_conversation(self, participants):
        """Return the conversation with exactly these participants."""
        return public(self._conversation(participants))

    def list_conversations(self, participant=None, query=None, pinned=None):
        """List conversations, optionally filtering by participant, text, and pin."""
        return public(
            [
                conversation
                for conversation in self.conversations
                if (
                    participant is None
                    or participant in conversation["participants"]
                )
                and (
                    query is None
                    or keyword_match(
                        query,
                        conversation["participants"],
                        conversation.get("title"),
                        conversation.get("pin_comment"),
                    )
                )
                and (pinned is None or conversation.get("pinned", False) == pinned)
            ]
        )

    def pin_conversation(self, participants, comment):
        """Pin a conversation for the Instance owner's attention."""
        conversation = self._conversation(participants)
        conversation["pinned"] = True
        conversation["pin_comment"] = required_text(comment, "comment")
        self.store.save()
        return public(conversation)

    def create_message(self, sender, recipients, content, timestamp):
        """Create a message inside an existing conversation."""
        recipients = list(dict.fromkeys(recipients))
        conversation = self._conversation([sender, *recipients])
        if any(
            item["sender"] == sender
            and self._same_people(item["recipients"], recipients)
            and item["timestamp"] == timestamp
            for item in self.messages
        ):
            raise ValueError(
                "A message with the same sender, recipients, and timestamp exists"
            )

        message = {
            "message_id": next_id(self.messages, "message_id", "message"),
            "conversation_id": conversation["conversation_id"],
            "sender": sender,
            "recipients": recipients,
            "content": content,
            "timestamp": timestamp,
        }
        self.messages.append(message)
        self.store.save()
        return public(message)

    def send_message(self, recipients, content, timestamp):
        """Send an outgoing message as the configured Instance owner."""
        if not self.owner:
            raise ValueError("Messaging owner is not configured")
        return self.create_message(self.owner, recipients, content, timestamp)

    def get_message(self, sender, recipients, timestamp):
        """Return a message selected by sender, recipients, and timestamp."""
        return public(self._message(sender, recipients, timestamp))

    def list_messages(
        self,
        participants=None,
        sender=None,
        start_time=None,
        end_time=None,
    ):
        """List messages with optional conversation, sender, and time filters."""
        results = []
        for message in self.messages:
            message_people = [message["sender"], *message["recipients"]]
            if (
                (participants is None or self._same_people(message_people, participants))
                and (sender is None or message["sender"] == sender)
                and in_time_range(message["timestamp"], start_time, end_time)
            ):
                results.append(message)
        return public(sorted(results, key=lambda item: parse_time(item["timestamp"])))

    def search_messages(self, query, participants=None, sender=None):
        """Search message participants and content with optional filters."""
        return [
            message
            for message in self.list_messages(
                participants=participants,
                sender=sender,
            )
            if keyword_match(
                query,
                message["sender"],
                message["recipients"],
                message["content"],
            )
        ]

    def update_message(self, sender, recipients, timestamp, content):
        """Edit the content of a selected message."""
        message = self._message(sender, recipients, timestamp)
        message["content"] = content
        self.store.save()
        return public(message)

    def create_draft(self, recipients, content, timestamp):
        """Compose a draft as the configured owner in an existing conversation."""
        if not self.owner:
            raise ValueError("Messaging owner is not configured")
        recipients = list(dict.fromkeys(recipients))
        conversation = self._conversation([self.owner, *recipients])
        if any(
            item["sender"] == self.owner
            and self._same_people(item["recipients"], recipients)
            and item["timestamp"] == timestamp
            for item in self.drafts
        ):
            raise ValueError(
                "A draft with the same sender, recipients, and timestamp exists"
            )
        draft = {
            "draft_id": next_id(self.drafts, "draft_id", "draft"),
            "conversation_id": conversation["conversation_id"],
            "sender": self.owner,
            "recipients": recipients,
            "content": content,
            "timestamp": timestamp,
        }
        self.drafts.append(draft)
        self.store.save()
        return public(draft)

    def list_drafts(self, participants=None, query=None):
        """List drafts with optional conversation and content filters."""
        return public(
            [
                draft
                for draft in self.drafts
                if (
                    participants is None
                    or self._same_people(
                        [draft["sender"], *draft["recipients"]], participants
                    )
                )
                and (
                    query is None
                    or keyword_match(
                        query,
                        draft["sender"],
                        draft["recipients"],
                        draft["content"],
                    )
                )
            ]
        )

    def update_draft(self, recipients, timestamp, content):
        """Replace the configured owner's selected message draft content."""
        if not self.owner:
            raise ValueError("Messaging owner is not configured")
        draft = self._draft(self.owner, recipients, timestamp)
        draft["content"] = content
        self.store.save()
        return public(draft)

"""Calendar events, reminders, filters, and conflict detection."""

from ._store import (
    JsonStore,
    in_time_range,
    keyword_match,
    next_id,
    parse_time,
    public,
    required_text,
)


class Calendar:
    """Store calendar events, event drafts, and reminders in a JSON file."""

    def __init__(self, state_path):
        self.store = JsonStore(
            state_path, {"events": [], "event_drafts": [], "reminders": []}
        )
        self.events = self.store.data.setdefault("events", [])
        self.event_drafts = self.store.data.setdefault("event_drafts", [])
        self.reminders = self.store.data.setdefault("reminders", [])

    @staticmethod
    def _check_event(start_time, end_time):
        if parse_time(start_time) >= parse_time(end_time):
            raise ValueError("start_time must be earlier than end_time")

    @staticmethod
    def _check_reminder(remind_time, due_time):
        if due_time is not None and parse_time(remind_time) > parse_time(due_time):
            raise ValueError("remind_time must not be later than due_time")

    def _event(self, title, start_time):
        """Find an event by title and start time."""
        for event in self.events:
            if event["title"] == title and event["start_time"] == start_time:
                return event
        raise KeyError(f"Event not found: {title}, {start_time}")

    def _event_draft(self, title, start_time):
        """Find an event draft by title and start time."""
        for draft in self.event_drafts:
            if draft["title"] == title and draft["start_time"] == start_time:
                return draft
        raise KeyError(f"Event draft not found: {title}, {start_time}")

    def _reminder(self, title, remind_time):
        """Find a reminder by title and reminder time."""
        for reminder in self.reminders:
            if reminder["title"] == title and reminder["remind_time"] == remind_time:
                return reminder
        raise KeyError(f"Reminder not found: {title}, {remind_time}")

    def create_event(
        self,
        title,
        start_time,
        end_time,
        description=None,
        location=None,
        attendees=None,
        organizer=None,
        status=None,
    ):
        """Create and save an event."""
        self._check_event(start_time, end_time)
        if any(
            item["title"] == title and item["start_time"] == start_time
            for item in self.events
        ):
            raise ValueError("An event with the same title and start_time exists")
        event = {
            "event_id": next_id(self.events, "event_id", "event"),
            "title": title,
            "start_time": start_time,
            "end_time": end_time,
        }
        optional = {
            "description": description,
            "location": location,
            "attendees": attendees,
            "organizer": organizer,
            "status": status,
        }
        event.update({key: value for key, value in optional.items() if value is not None})
        self.events.append(event)
        self.store.save()
        return public(event)

    def get_event(self, title, start_time):
        """Return an event selected by title and start time."""
        return public(self._event(title, start_time))

    def list_events(
        self,
        query=None,
        start_time=None,
        end_time=None,
        status=None,
        attendee=None,
    ):
        """Search and filter events; time filters use interval overlap."""
        results = []
        for event in self.events:
            overlaps = (
                start_time is None or parse_time(event["end_time"]) > parse_time(start_time)
            ) and (
                end_time is None or parse_time(event["start_time"]) < parse_time(end_time)
            )
            searchable = keyword_match(
                query or "",
                event["title"],
                event.get("description"),
                event.get("location"),
                event.get("organizer"),
                event.get("attendees", []),
            )
            if (
                overlaps
                and (query is None or searchable)
                and (status is None or event.get("status") == status)
                and (attendee is None or attendee in event.get("attendees", []))
            ):
                results.append(event)
        return public(results)

    def create_event_draft(
        self,
        title,
        start_time,
        end_time,
        description=None,
        location=None,
        attendees=None,
        organizer=None,
        status=None,
    ):
        """Prepare and save an event draft without creating an event."""
        self._check_event(start_time, end_time)
        if any(
            item["title"] == title and item["start_time"] == start_time
            for item in self.event_drafts
        ):
            raise ValueError(
                "An event draft with the same title and start_time exists"
            )
        draft = {
            "event_draft_id": next_id(
                self.event_drafts, "event_draft_id", "event_draft"
            ),
            "title": title,
            "start_time": start_time,
            "end_time": end_time,
        }
        optional = {
            "description": description,
            "location": location,
            "attendees": attendees,
            "organizer": organizer,
            "status": status,
        }
        draft.update(
            {key: value for key, value in optional.items() if value is not None}
        )
        self.event_drafts.append(draft)
        self.store.save()
        return public(draft)

    def get_event_draft(self, title, start_time):
        """Return an event draft selected by title and start time."""
        return public(self._event_draft(title, start_time))

    def list_event_drafts(
        self, query=None, start_time=None, end_time=None, attendee=None
    ):
        """Search and filter event drafts; time filters use interval overlap."""
        return public(
            [
                draft
                for draft in self.event_drafts
                if (
                    start_time is None
                    or parse_time(draft["end_time"]) > parse_time(start_time)
                )
                and (
                    end_time is None
                    or parse_time(draft["start_time"]) < parse_time(end_time)
                )
                and (
                    query is None
                    or keyword_match(
                        query,
                        draft["title"],
                        draft.get("description"),
                        draft.get("location"),
                        draft.get("organizer"),
                        draft.get("attendees", []),
                    )
                )
                and (
                    attendee is None or attendee in draft.get("attendees", [])
                )
            ]
        )

    def update_event_draft(self, title, start_time, **changes):
        """Update an event draft selected by title and start time."""
        draft = self._event_draft(title, start_time)
        changes.pop("event_draft_id", None)
        updated = {**draft, **changes}
        self._check_event(updated["start_time"], updated["end_time"])
        if any(
            item is not draft
            and item["title"] == updated["title"]
            and item["start_time"] == updated["start_time"]
            for item in self.event_drafts
        ):
            raise ValueError(
                "An event draft with the same title and start_time exists"
            )
        draft.update(changes)
        self.store.save()
        return public(draft)

    def delete_event_draft(self, title, start_time):
        """Delete an event draft without affecting calendar events."""
        draft = self._event_draft(title, start_time)
        self.event_drafts.remove(draft)
        self.store.save()
        return public(draft)

    def update_event(self, title, start_time, **changes):
        """Update an event selected by title and start time."""
        event = self._event(title, start_time)
        changes.pop("event_id", None)
        updated = {**event, **changes}
        self._check_event(updated["start_time"], updated["end_time"])
        if any(
            item is not event
            and item["title"] == updated["title"]
            and item["start_time"] == updated["start_time"]
            for item in self.events
        ):
            raise ValueError("An event with the same title and start_time exists")
        event.update(changes)
        self.store.save()
        return public(event)

    def delete_event(self, title, start_time):
        """Delete an event selected by title and start time."""
        event = self._event(title, start_time)
        self.events.remove(event)
        self.store.save()
        return public(event)

    def find_conflicts(self, start_time, end_time, include_cancelled=False):
        """Return events that overlap a proposed time interval."""
        self._check_event(start_time, end_time)
        return public(
            [
                event
                for event in self.events
                if (include_cancelled or event.get("status") != "cancelled")
                and parse_time(event["start_time"]) < parse_time(end_time)
                and parse_time(event["end_time"]) > parse_time(start_time)
            ]
        )

    def create_reminder(
        self,
        title,
        remind_time,
        content,
        due_time=None,
        completed=False,
    ):
        """Create and save a reminder."""
        self._check_reminder(remind_time, due_time)
        if any(
            item["title"] == title and item["remind_time"] == remind_time
            for item in self.reminders
        ):
            raise ValueError("A reminder with the same title and remind_time exists")
        reminder = {
            "reminder_id": next_id(self.reminders, "reminder_id", "reminder"),
            "title": title,
            "remind_time": remind_time,
            "content": required_text(content, "content"),
            "completed": completed,
        }
        if due_time is not None:
            reminder["due_time"] = due_time
        self.reminders.append(reminder)
        self.store.save()
        return public(reminder)

    def get_reminder(self, title, remind_time):
        """Return a reminder selected by title and reminder time."""
        return public(self._reminder(title, remind_time))

    def list_reminders(
        self, query=None, start_time=None, end_time=None, completed=None
    ):
        """Search and filter reminders."""
        return public(
            [
                reminder
                for reminder in self.reminders
                if (completed is None or reminder["completed"] == completed)
                and in_time_range(reminder["remind_time"], start_time, end_time)
                and (
                    query is None
                    or keyword_match(
                        query, reminder["title"], reminder["content"]
                    )
                )
            ]
        )

    def update_reminder(self, title, remind_time, **changes):
        """Update a reminder selected by title and reminder time."""
        reminder = self._reminder(title, remind_time)
        changes.pop("reminder_id", None)
        updated = {**reminder, **changes}
        self._check_reminder(updated["remind_time"], updated.get("due_time"))
        if any(
            item is not reminder
            and item["title"] == updated["title"]
            and item["remind_time"] == updated["remind_time"]
            for item in self.reminders
        ):
            raise ValueError("A reminder with the same title and remind_time exists")
        reminder.update(changes)
        self.store.save()
        return public(reminder)

    def delete_reminder(self, title, remind_time):
        """Delete a reminder selected by title and reminder time."""
        reminder = self._reminder(title, remind_time)
        self.reminders.remove(reminder)
        self.store.save()
        return public(reminder)

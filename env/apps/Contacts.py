"""Contact storage and search."""

from ._store import JsonStore, keyword_match, next_id, public, required_text


class Contacts:
    """Store contacts in a JSON file and address them by name."""

    def __init__(self, state_path):
        self.store = JsonStore(state_path, {"contacts": []})
        self.contacts = self.store.data.setdefault("contacts", [])

    def _contact(self, name):
        """Find the first contact with an exact name match."""
        for contact in self.contacts:
            if contact["name"] == name:
                return contact
        raise KeyError(f"Contact not found: {name}")

    def create_contact(
        self,
        name,
        aliases=None,
        emails=None,
        phone_numbers=None,
        organization=None,
        relationship=None,
        notes=None,
    ):
        """Create and save a contact."""
        if any(item["name"] == name for item in self.contacts):
            raise ValueError(f"Contact already exists: {name}")
        contact = {
            "contact_id": next_id(self.contacts, "contact_id", "contact"),
            "name": name,
            "flagged": False,
        }
        optional = {
            "aliases": aliases,
            "emails": emails,
            "phone_numbers": phone_numbers,
            "organization": organization,
            "relationship": relationship,
            "notes": notes,
        }
        contact.update({key: value for key, value in optional.items() if value is not None})
        self.contacts.append(contact)
        self.store.save()
        return public(contact)

    def get_contact(self, name):
        """Return a contact selected by exact name."""
        return public(self._contact(name))

    def list_contacts(self, flagged=None):
        """Return contacts, optionally filtered by attention flag."""
        return public(
            [
                contact
                for contact in self.contacts
                if flagged is None or contact.get("flagged", False) == flagged
            ]
        )

    def search_contacts(self, query):
        """Search all user-visible contact fields."""
        return public(
            [
                contact
                for contact in self.contacts
                if keyword_match(
                    query,
                    contact["name"],
                    contact.get("aliases", []),
                    contact.get("emails", []),
                    contact.get("phone_numbers", []),
                    contact.get("organization"),
                    contact.get("relationship"),
                    contact.get("notes"),
                    contact.get("flag_comment"),
                )
            ]
        )

    def update_contact(self, name, **changes):
        """Update a contact selected by exact name."""
        contact = self._contact(name)
        changes.pop("contact_id", None)
        new_name = changes.get("name", name)
        if any(item is not contact and item["name"] == new_name for item in self.contacts):
            raise ValueError(f"Contact already exists: {new_name}")
        contact.update(changes)
        self.store.save()
        return public(contact)

    def flag_contact(self, name, comment):
        """Flag a contact for the Instance owner's attention."""
        contact = self._contact(name)
        contact["flagged"] = True
        contact["flag_comment"] = required_text(comment, "comment")
        self.store.save()
        return public(contact)

    def delete_contact(self, name):
        """Delete a contact selected by exact name."""
        contact = self._contact(name)
        self.contacts.remove(contact)
        self.store.save()
        return public(contact)

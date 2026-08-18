"""A single personal wallet with payments, deposits, and transaction history."""

from ._store import (
    JsonStore,
    clone,
    in_time_range,
    keyword_match,
    next_id,
    parse_time,
    public,
    required_text,
)


class Wallet:
    """Manage one wallet and its transaction history in a JSON file."""

    STATUSES = {"pending", "completed", "failed", "refunded"}
    TRANSACTION_TYPES = {"payment", "deposit", "refund"}

    def __init__(self, state_path):
        self.store = JsonStore(
            state_path, {"wallet": None, "transactions": [], "drafts": []}
        )
        self.transactions = self.store.data.setdefault("transactions", [])
        self.drafts = self.store.data.setdefault("drafts", [])

    def _wallet(self):
        wallet = self.store.data.get("wallet")
        if wallet is None:
            raise ValueError("Wallet has not been set up")
        return wallet

    def _transaction(self, transaction_id):
        for transaction in self.transactions:
            if transaction["transaction_id"] == transaction_id:
                return transaction
        raise KeyError(f"Transaction not found: {transaction_id}")

    def _draft(self, timestamp):
        for draft in self.drafts:
            if draft["timestamp"] == timestamp:
                return draft
        raise KeyError(f"Payment draft not found: {timestamp}")

    @classmethod
    def _check_status(cls, status, allow_refunded=False):
        """Validate the small status vocabulary used by the wallet."""
        allowed = cls.STATUSES if allow_refunded else cls.STATUSES - {"refunded"}
        if status not in allowed:
            raise ValueError(f"status must be one of: {', '.join(sorted(allowed))}")

    @staticmethod
    def _check_amount(amount):
        if amount <= 0:
            raise ValueError("amount must be greater than zero")

    @staticmethod
    def _balance_effect(transaction, status=None):
        """Return how a transaction status contributes to the current balance."""
        status = status or transaction["status"]
        transaction_type = transaction["transaction_type"]
        if transaction_type == "payment" and status in {"completed", "refunded"}:
            return -transaction["amount"]
        if transaction_type in {"deposit", "refund"} and status == "completed":
            return transaction["amount"]
        return 0

    def _save_transaction(self, transaction):
        """Append a transaction and apply its completed balance effect."""
        wallet = self._wallet()
        new_balance = wallet["balance"] + self._balance_effect(transaction)
        if new_balance < 0:
            raise ValueError("Insufficient balance")
        wallet["balance"] = new_balance
        self.transactions.append(transaction)
        self.store.save()
        return clone(transaction)

    def setup_wallet(
        self,
        currency,
        balance=0,
        account_type="personal",
        institution=None,
    ):
        """Create the single wallet."""
        if self.store.data.get("wallet") is not None:
            raise ValueError("Wallet has already been set up")
        if balance < 0:
            raise ValueError("balance cannot be negative")

        wallet = {
            "wallet_id": "wallet_000001",
            "currency": currency,
            "balance": balance,
            "account_type": account_type,
        }
        if institution is not None:
            wallet["institution"] = institution
        self.store.data["wallet"] = wallet
        self.store.save()
        return clone(wallet)

    def get_wallet(self):
        """Return wallet details without transaction rows."""
        return clone(self._wallet())

    def update_wallet(self, account_type=None, institution=None):
        """Update the wallet's small set of editable account details."""
        wallet = self._wallet()
        if account_type is not None:
            wallet["account_type"] = account_type
        if institution is not None:
            wallet["institution"] = institution
        self.store.save()
        return clone(wallet)

    def get_transaction(self, transaction_id):
        """Return one transaction selected by transaction ID."""
        return clone(self._transaction(transaction_id))

    def add_funds(
        self,
        amount,
        timestamp,
        source="manual",
        status="completed",
        reference=None,
        description=None,
    ):
        """Record incoming money and increase balance when completed."""
        self._check_amount(amount)
        self._check_status(status)
        wallet = self._wallet()
        transaction = {
            "transaction_id": next_id(
                self.transactions, "transaction_id", "transaction"
            ),
            "transaction_type": "deposit",
            "source": source,
            "amount": amount,
            "currency": wallet["currency"],
            "timestamp": timestamp,
            "status": status,
            "flagged": False,
        }
        if reference is not None:
            transaction["reference"] = reference
        if description is not None:
            transaction["description"] = description
        return self._save_transaction(transaction)

    def pay(
        self,
        merchant,
        amount,
        timestamp,
        status="completed",
        category=None,
        reference=None,
        description=None,
    ):
        """Record a payment and decrease balance when completed."""
        self._check_amount(amount)
        self._check_status(status)
        wallet = self._wallet()
        transaction = {
            "transaction_id": next_id(
                self.transactions, "transaction_id", "transaction"
            ),
            "transaction_type": "payment",
            "merchant": merchant,
            "amount": amount,
            "currency": wallet["currency"],
            "timestamp": timestamp,
            "status": status,
            "flagged": False,
        }
        optional = {
            "category": category,
            "reference": reference,
            "description": description,
        }
        transaction.update(
            {key: value for key, value in optional.items() if value is not None}
        )
        return self._save_transaction(transaction)

    def create_draft(
        self,
        merchant,
        amount,
        timestamp,
        category=None,
        reference=None,
        description=None,
    ):
        """Prepare a payment without creating a transaction or changing balance."""
        self._check_amount(amount)
        wallet = self._wallet()
        if any(draft["timestamp"] == timestamp for draft in self.drafts):
            raise ValueError("A payment draft with the same timestamp exists")
        draft = {
            "draft_id": next_id(self.drafts, "draft_id", "draft"),
            "merchant": merchant,
            "amount": amount,
            "currency": wallet["currency"],
            "timestamp": timestamp,
        }
        optional = {
            "category": category,
            "reference": reference,
            "description": description,
        }
        draft.update(
            {key: value for key, value in optional.items() if value is not None}
        )
        self.drafts.append(draft)
        self.store.save()
        return public(draft)

    def list_drafts(self, query=None, min_amount=None, max_amount=None):
        """List payment drafts with optional text and amount filters."""
        return public(
            [
                draft
                for draft in self.drafts
                if (query is None or self._transaction_matches(query, draft))
                and (min_amount is None or draft["amount"] >= min_amount)
                and (max_amount is None or draft["amount"] <= max_amount)
            ]
        )

    def update_draft(self, timestamp, **changes):
        """Update a payment draft without changing its creation time."""
        draft = self._draft(timestamp)
        changes.pop("draft_id", None)
        changes.pop("currency", None)
        changes.pop("timestamp", None)
        if "amount" in changes:
            self._check_amount(changes["amount"])
        draft.update(changes)
        self.store.save()
        return public(draft)

    def list_transactions(
        self,
        query=None,
        transaction_type=None,
        status=None,
        start_time=None,
        end_time=None,
        min_amount=None,
        max_amount=None,
        flagged=None,
    ):
        """Search and filter transaction history in timestamp order."""
        results = [
            transaction
            for transaction in self.transactions
            if (query is None or self._transaction_matches(query, transaction))
            and (
                transaction_type is None
                or transaction["transaction_type"] == transaction_type
            )
            and (status is None or transaction["status"] == status)
            and in_time_range(transaction["timestamp"], start_time, end_time)
            and (min_amount is None or transaction["amount"] >= min_amount)
            and (max_amount is None or transaction["amount"] <= max_amount)
            and (flagged is None or transaction.get("flagged", False) == flagged)
        ]
        return clone(sorted(results, key=lambda item: parse_time(item["timestamp"])))

    def flag_transaction(self, transaction_id, comment):
        """Flag a transaction for the Instance owner's attention."""
        transaction = self._transaction(transaction_id)
        transaction["flagged"] = True
        transaction["flag_comment"] = required_text(comment, "comment")
        self.store.save()
        return clone(transaction)

    @staticmethod
    def _transaction_matches(query, transaction):
        """Match a query against the descriptive transaction fields."""
        return keyword_match(
            query,
            transaction.get("merchant"),
            transaction.get("source"),
            transaction.get("category"),
            transaction.get("reference"),
            transaction.get("description"),
            transaction.get("flag_comment"),
        )

    def update_transaction_status(self, transaction_id, status):
        """Move a transaction between pending, completed, and failed states."""
        self._check_status(status)
        transaction = self._transaction(transaction_id)
        if transaction["transaction_type"] == "refund":
            raise ValueError("Refund transaction status cannot be changed")
        if transaction["status"] == "refunded":
            raise ValueError("A refunded payment is final")
        wallet = self._wallet()
        old_effect = self._balance_effect(transaction)
        new_effect = self._balance_effect(transaction, status)
        new_balance = wallet["balance"] - old_effect + new_effect
        if new_balance < 0:
            raise ValueError("Insufficient balance")
        wallet["balance"] = new_balance
        transaction["status"] = status
        self.store.save()
        return clone(transaction)

    def refund_transaction(self, transaction_id, timestamp, description=None):
        """Refund a completed payment and create a separate refund transaction."""
        payment = self._transaction(transaction_id)
        if payment["transaction_type"] != "payment":
            raise ValueError("Only payments can be refunded")
        if payment["status"] != "completed":
            raise ValueError("Only completed payments can be refunded")

        payment["status"] = "refunded"
        refund = {
            "transaction_id": next_id(
                self.transactions, "transaction_id", "transaction"
            ),
            "transaction_type": "refund",
            "merchant": payment["merchant"],
            "amount": payment["amount"],
            "currency": payment["currency"],
            "timestamp": timestamp,
            "status": "completed",
            "reference": transaction_id,
            "flagged": False,
        }
        if description is not None:
            refund["description"] = description
        return self._save_transaction(refund)

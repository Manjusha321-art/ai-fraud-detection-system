from collections import defaultdict, deque


class FeatureStore:
    def __init__(self, window_size=10):
        self.window_size = window_size

        # Stores recent transactions for each account
        self.history = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )

    def add_transaction(self, account_id, amount):
        """
        Add a transaction to an account's transaction history.
        """
        self.history[account_id].append(amount)

    def get_history(self, account_id):
        """
        Return the recent transaction history of an account.
        """
        return list(self.history[account_id])


if __name__ == "__main__":
    store = FeatureStore(window_size=3)

    store.add_transaction("ACC-101", 100)
    store.add_transaction("ACC-101", 120)
    store.add_transaction("ACC-101", 110)

    print(store.get_history("ACC-101"))
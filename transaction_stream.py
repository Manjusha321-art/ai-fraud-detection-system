import random
from datetime import datetime


class TransactionStream:
    def __init__(self, account_id):
        self.account_id = account_id

        self.normal_locations = [
            "Hyderabad",
            "Bangalore",
            "Chennai"
        ]

        self.normal_devices = [
            "DEVICE-001",
            "DEVICE-002"
        ]

    def generate_transaction(self):
        """
        Generate a normal transaction.
        """

        amount = random.randint(90, 130)

        location = random.choice(
            self.normal_locations
        )

        device_id = random.choice(
            self.normal_devices
        )

        current_hour = datetime.now().hour

        return {
            "account_id": self.account_id,
            "amount": amount,
            "location": location,
            "device_id": device_id,
            "hour": current_hour
        }

    def generate_suspicious_transaction(self):
        """
        Generate a suspicious transaction with
        unusual amount, location, and device.
        """

        amount = random.randint(800, 1200)

        location = "New York"

        device_id = "UNKNOWN-DEVICE"

        hour = random.choice([
            1,
            2,
            3,
            4
        ])

        return {
            "account_id": self.account_id,
            "amount": amount,
            "location": location,
            "device_id": device_id,
            "hour": hour
        }


if __name__ == "__main__":

    stream = TransactionStream("ACC-101")

    print("Normal transaction:")
    print(stream.generate_transaction())

    print("\nSuspicious transaction:")
    print(
        stream.generate_suspicious_transaction()
    )
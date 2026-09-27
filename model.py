import statistics


class AnomalyDetector:

    def __init__(
        self,
        z_score_threshold=2.5,
        known_locations=None,
        known_devices=None
    ):

        self.z_score_threshold = z_score_threshold

        self.known_locations = (
            known_locations
            if known_locations
            else []
        )

        self.known_devices = (
            known_devices
            if known_devices
            else []
        )

    def calculate_z_score(
        self,
        amount,
        history
    ):

        if len(history) < 2:
            return 0.0

        mean = statistics.mean(history)

        standard_deviation = (
            statistics.stdev(history)
        )

        if standard_deviation == 0:
            return 0.0

        z_score = (
            amount - mean
        ) / standard_deviation

        return z_score

    def evaluate_transaction(
        self,
        transaction,
        history
    ):

        amount = transaction["amount"]

        location = transaction["location"]

        device_id = transaction["device_id"]

        hour = transaction["hour"]

        z_score = self.calculate_z_score(
            amount,
            history
        )

        risk_score = 0

        reasons = []

        if abs(z_score) >= self.z_score_threshold:

            risk_score += 40

            reasons.append(
                "Unusual transaction amount"
            )

        if location not in self.known_locations:

            risk_score += 20

            reasons.append(
                "Unusual location"
            )

        if device_id not in self.known_devices:

            risk_score += 25

            reasons.append(
                "Unknown device"
            )

        if hour < 6 or hour > 23:

            risk_score += 15

            reasons.append(
                "Unusual transaction time"
            )

        is_anomaly = (
            risk_score >= 50
        )

        return {

            "is_anomaly": is_anomaly,

            "risk_score": risk_score,

            "z_score": round(
                z_score,
                2
            ),

            "reasons": reasons

        }


if __name__ == "__main__":

    detector = AnomalyDetector(

        z_score_threshold=2.5,

        known_locations=[

            "Hyderabad",

            "Bangalore",

            "Chennai"

        ],

        known_devices=[

            "DEVICE-001",

            "DEVICE-002"

        ]

    )

    history = [

        100,

        120,

        110,

        105,

        95

    ]

    transaction = {

        "amount": 950,

        "location": "New York",

        "device_id": "UNKNOWN-DEVICE",

        "hour": 2

    }

    result = detector.evaluate_transaction(

        transaction,

        history

    )

    print(result)
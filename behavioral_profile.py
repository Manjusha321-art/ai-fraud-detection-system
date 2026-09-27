class BehavioralProfile:

    def __init__(self):

        self.transaction_history = []


    def add_transaction(
        self,
        transaction
    ):

        self.transaction_history.append(
            transaction
        )


    def get_profile(self):

        if not self.transaction_history:

            return {

                "average_amount": 0,

                "maximum_amount": 0,

                "known_locations": [],

                "known_devices": [],

                "usual_hours": []

            }


        amounts = [

            transaction["amount"]

            for transaction in self.transaction_history

        ]


        locations = [

            transaction["location"]

            for transaction in self.transaction_history

        ]


        devices = [

            transaction["device_id"]

            for transaction in self.transaction_history

        ]


        hours = [

            transaction["hour"]

            for transaction in self.transaction_history

        ]


        return {

            "average_amount": round(

                sum(amounts)

                / len(amounts),

                2

            ),

            "maximum_amount": max(

                amounts

            ),

            "known_locations": list(

                set(locations)

            ),

            "known_devices": list(

                set(devices)

            ),

            "usual_hours": list(

                set(hours)

            )

        }


    def analyze_behavior(

        self,

        transaction

    ):


        profile = self.get_profile()


        deviations = []


        # Amount deviation
        if profile["average_amount"] > 0:

            if transaction["amount"] > (

                profile["average_amount"]

                * 3

            ):

                deviations.append(

                    "Transaction amount is significantly higher than usual"

                )


        # Location deviation
        if transaction["location"] not in (

            profile["known_locations"]

        ):

            deviations.append(

                "Transaction location is new for this account"

            )


        # Device deviation
        if transaction["device_id"] not in (

            profile["known_devices"]

        ):

            deviations.append(

                "Transaction device is new for this account"

            )


        # Time deviation
        if transaction["hour"] not in (

            profile["usual_hours"]

        ):

            deviations.append(

                "Transaction time is outside usual behavior"

            )


        return {

            "profile": profile,

            "deviations": deviations

        }
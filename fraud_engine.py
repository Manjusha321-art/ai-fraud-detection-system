import joblib
import pandas as pd

from src.behavioral_profile import BehavioralProfile


class FraudEngine:

    def __init__(self):

        self.model = joblib.load(
            "fraud_model.pkl"
        )

        self.scaler = joblib.load(
            "amount_scaler.pkl"
        )

        self.behavioral_profile = BehavioralProfile()

        print(
            "Fraud Detection Engine Loaded"
        )


    def predict_with_ml(self, transaction):
   
        if "Time" not in transaction and "time" in transaction:

            transaction["Time"] = transaction["time"]


        if "Amount" not in transaction and "amount" in transaction:
           

           transaction["Amount"] = transaction["amount"]

        feature_order = [

            "Time",

            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
            "V7",
            "V8",
            "V9",
            "V10",
            "V11",
            "V12",
            "V13",
            "V14",
            "V15",
            "V16",
            "V17",
            "V18",
            "V19",
            "V20",
            "V21",
            "V22",
            "V23",
            "V24",
            "V25",
            "V26",
            "V27",
            "V28",

            "Amount"

        ]


        transaction_df = pd.DataFrame(

            [

                [

                    transaction[feature]

                    for feature in feature_order

                ]

            ],

            columns=feature_order

        )


        transaction_df["Amount"] = (

            self.scaler.transform(

                transaction_df[["Amount"]]

            ).ravel()

        )


        fraud_probability = (

            self.model.predict_proba(

                transaction_df

            )[0][1]

        )


        ml_threshold = 0.3


        return {

            "is_fraud": bool(

                fraud_probability >= ml_threshold

            ),

            "fraud_probability": round(

                float(

                    fraud_probability

                ),

                4

            )

        }


    def analyze_transaction(

        self,

        transaction

    ):


        # Analyze the account's behavior
        behavior_result = (

            self.behavioral_profile.analyze_behavior(

                transaction

            )

        )


        # Machine learning prediction
        ml_result = (

            self.predict_with_ml(

                transaction

            )

        )


        risk_score = 0

        reasons = []


        # Rule 1: High transaction amount
        if transaction["amount"] > 1000:

            risk_score += 25

            reasons.append(

                "Unusually high transaction amount"

            )


        # Rule 2: Unknown device
        if transaction.get(

            "device_id"

        ) == "UNKNOWN-DEVICE":

            risk_score += 25

            reasons.append(

                "Unknown device"

            )


        # Rule 3: Unusual location
        if transaction.get(

            "location"

        ) not in [

            "Hyderabad",

            "Bangalore",

            "Chennai"

        ]:

            risk_score += 25

            reasons.append(

                "Unusual location"

            )


        # Rule 4: Unusual transaction time
        if transaction.get(

            "hour"

        ) < 6:

            risk_score += 25

            reasons.append(

                "Unusual transaction time"

            )


        # Rule 5: ML model prediction
        if ml_result["is_fraud"]:

            risk_score += 25

            reasons.append(

                "Machine learning model detected suspicious behavior"

            )


        # Add behavioral deviations
        for deviation in (

            behavior_result["deviations"]

        ):

            if deviation not in reasons:

                reasons.append(

                    deviation

                )


        # Final fraud decision
        is_fraud = (

            risk_score >= 50

        )


        # Add transaction to the behavioral history
        self.behavioral_profile.add_transaction(

            transaction

        )


        return {

            "account_id": transaction[

                "account_id"

            ],

            "amount": transaction[

                "amount"

            ],

            "risk_score": risk_score,

            "is_fraud": is_fraud,

            "fraud_probability": ml_result[

                "fraud_probability"

            ],

            "reasons": reasons,

            "behavioral_profile": behavior_result[

                "profile"

            ],

            "behavioral_deviations": behavior_result[

                "deviations"

            ]

        }
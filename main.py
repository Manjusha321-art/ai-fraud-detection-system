import time
import random

from src.fraud_engine import FraudEngine
from src.dispatcher import AlertDispatcher
from src.transaction_stream import TransactionStream


def add_ml_features(transaction):

    transaction["Time"] = random.randint(
        0,
        172800
    )

    for i in range(1, 29):

        transaction[f"V{i}"] = random.uniform(
            -2,
            2
        )

    return transaction


def process_transaction(
    transaction,
    engine,
    dispatcher
):

    account_id = transaction["account_id"]

    amount = transaction["amount"]


    # Add ML features
    transaction = add_ml_features(
        transaction
    )


    # Convert amount for the ML model
    transaction["Amount"] = amount


    # Analyze transaction
    result = engine.analyze_transaction(
        transaction
    )


    print(
        "\n--- Transaction Analysis ---"
    )

    print(
        f"Account ID: {account_id}"
    )

    print(
        f"Transaction Amount: ₹{amount}"
    )

    print(
        f"Location: {transaction['location']}"
    )

    print(
        f"Device ID: {transaction['device_id']}"
    )

    print(
        f"Transaction Hour: {transaction['hour']}"
    )

    print(
        f"Risk Score: {result['risk_score']}"
    )

    print(
        f"ML Fraud Probability: "
        f"{result['fraud_probability']}"
    )

    print(
        f"Fraud Detected: "
        f"{result['is_fraud']}"
    )


    if result["is_fraud"]:

        print(
            "🚨 FRAUD ALERT 🚨"
        )

        print(
            "Action: Transaction requires investigation."
        )

        print(
            "Reasons:"
        )

        for reason in result["reasons"]:

            print(
                f"- {reason}"
            )


        dispatcher.send_alert(

            account_id=account_id,

            amount=amount,

            z_score=0

        )


    else:

        print(
            "✅ Transaction appears normal."
        )


if __name__ == "__main__":


    engine = FraudEngine()


    dispatcher = AlertDispatcher()


    stream = TransactionStream(
        "ACC-101"
    )


    print(
        "🚀 Real-Time Fraud Detection System Started"
    )


    for _ in range(5):

        transaction = (

            stream.generate_transaction()

        )


        process_transaction(

            transaction,

            engine,

            dispatcher

        )


        time.sleep(1)


    suspicious_transaction = (

        stream.generate_suspicious_transaction()

    )


    process_transaction(

        suspicious_transaction,

        engine,

        dispatcher

    )
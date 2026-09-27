import pandas as pd
import joblib


class MLPredictor:

    def __init__(self):

        # Load the trained Random Forest model
        self.model = joblib.load(
            "fraud_model.pkl"
        )

        # Load the scaler used during training
        self.scaler = joblib.load(
            "amount_scaler.pkl"
        )

        print(
            "ML model loaded successfully"
        )


    def predict(self, transaction):

        # Convert transaction dictionary into a DataFrame
        transaction_df = pd.DataFrame(
            [transaction]
        )


        # Scale Amount using the same scaler
        # used during model training
        transaction_df["Amount"] = self.scaler.transform(
            transaction_df[["Amount"]]
        )


        # Get fraud probability
        fraud_probability = self.model.predict_proba(
            transaction_df
        )[0][1]


        # Use the same threshold used during training
        threshold = 0.3


        # Decide whether transaction is fraud
        is_fraud = (
            fraud_probability >= threshold
        )


        return {

            "is_fraud": is_fraud,

            "fraud_probability": round(
                fraud_probability,
                4
            )

        }


if __name__ == "__main__":

    predictor = MLPredictor()


    # Test transaction
    transaction = {

    "Time": 406.0,

    "V1": -2.3122265423263,

    "V2": 1.95199201064158,

    "V3": -1.60985073229769,

    "V4": 3.9979055875468,

    "V5": -0.522187864667764,

    "V6": -1.42654531920595,

    "V7": -2.53738730624579,

    "V8": 1.39165724829804,

    "V9": -2.77008927719433,

    "V10": -2.77227214465915,

    "V11": 3.20203320709635,

    "V12": -2.89990738849473,

    "V13": -0.595221881324605,

    "V14": -4.28925378244217,

    "V15": 0.389724120274487,

    "V16": -1.14074717980657,

    "V17": -2.83005567450437,

    "V18": -0.0168224681808257,

    "V19": 0.416955705037907,

    "V20": 0.126910559061474,

    "V21": 0.517232370861764,

    "V22": -0.0350493686052974,

    "V23": -0.465211076182388,

    "V24": 0.320198198514526,

    "V25": 0.0445191674731724,

    "V26": 0.177839798284401,

    "V27": 0.261145002567677,

    "V28": -0.143275874698919,

    "Amount": 0.0

}
        

    


    result = predictor.predict(
        transaction
    )


    print(
        result
    )
class AlertDispatcher:

    def send_alert(
        self,
        account_id,
        amount,
        z_score
    ):

        print(
            "\n📢 Alert dispatched successfully."
        )

        print(
            f"Account ID: {account_id}"
        )

        print(
            f"Transaction Amount: ₹{amount}"
        )

        print(
            "Action: Transaction requires investigation."
        )
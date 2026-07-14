from monitoring.storage.retraining_history import (
    can_trigger_retraining,
)


def main():

    print(
        can_trigger_retraining()
    )


if __name__ == "__main__":
    main()
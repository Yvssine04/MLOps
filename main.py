import argparse

from model_pipeline import (
    load_data,
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


def main():
    parser = argparse.ArgumentParser(description="Pipeline ML churn")
    parser.add_argument("--load_data", action="store_true", help="Load the CSV file")
    parser.add_argument("--prepare", action="store_true", help="Prepare the data")
    parser.add_argument("--train", action="store_true", help="Train and save the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the saved model")
    parser.add_argument("--load", action="store_true", help="Load the saved model")
    parser.add_argument("--all", action="store_true", help="Run the full pipeline")
    args = parser.parse_args()

    if args.all:
        args.load_data = True
        args.prepare = True
        args.train = True
        args.load = True
        args.evaluate = True

    if args.load_data:
        load_data()

    if args.prepare:
        prepare_data()

    if args.train:
        x_train, x_test, y_train, y_test = prepare_data()
        model = train_model(x_train, y_train)
        save_model(model)

    if args.evaluate:
        x_train, x_test, y_train, y_test = prepare_data()
        model = load_model()
        evaluate_model(model, x_test, y_test)

    if args.load:
        load_model()

    if args.all:
        print("Full pipeline done.")


if __name__ == "__main__":
    main()

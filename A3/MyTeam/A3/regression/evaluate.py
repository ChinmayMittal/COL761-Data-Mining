import argparse


def main():
    parser = argparse.ArgumentParser(description="Evaluating the classification model")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    args = parser.parse_args()
    print(f"Evaluating the classification model. Model will be loaded from {args.model_path}. Test dataset will be loaded from {args.dataset_path}.")


if __name__=="__main__":
    main()

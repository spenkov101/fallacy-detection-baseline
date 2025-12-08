import jsonlines
import pandas as pd

def load_dataset(path):
    """
    Load a .jsonl dataset into a pandas DataFrame.
    Expected fields: "text" and "label"
    """
    with jsonlines.open(path, 'r') as reader:
        data = [item for item in reader]
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = load_dataset("data/sample_examples.jsonl")
    print(df.head())

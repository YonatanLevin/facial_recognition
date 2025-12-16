from torch import manual_seed

from src.pipeline import Pipeline

def main():
    manual_seed(42)
    pipeline = Pipeline()

if __name__ == '__main__':
    main()
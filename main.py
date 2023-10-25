import pandas as pd
from learning import *


if __name__ == "__main__":
    train_data = pd.read_csv('Dataset/train.csv', index_col=0)
    test_data = pd.read_csv('Dataset/test.csv', index_col=0)
    learn = Learning_data(train_data, test_data, 6000, 0.1, 0.01)
    learn.learn_data('submit')

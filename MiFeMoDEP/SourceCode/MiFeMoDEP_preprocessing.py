import pandas as pd

def preprocessing():
    path_to_source_code_random = './DefectorsDataset/defectors/line_bug_prediction_splits/random'

    train_source_code_df = pd.read_parquet(f'{path_to_source_code_random}/train.parquet.gzip')
    train_source_code_df = train_source_code_df.reset_index(drop=True)

    test_source_code_df = pd.read_parquet(f'{path_to_source_code_random}/test.parquet.gzip')
    test_source_code_df = test_source_code_df.reset_index(drop=True)

    # compute file-level labels
    train_source_code_df['target'] = train_source_code_df['lines'].apply(lambda line : 0 if len(line) == 0 else 1)
    test_source_code_df['target'] = test_source_code_df['lines'].apply(lambda line : 0 if len(line) == 0 else 1)

    # decode the source code properly since CSV storage may have errors like type/decoding conversion issues sometimes
    train_source_code_df['content'] = train_source_code_df['content'].apply(lambda x : '' if x is None else x.decode("latin-1"))
    test_source_code_df['content'] = test_source_code_df['content'].apply(lambda x : '' if x is None else x.decode("latin-1"))
    train_source_code_df.to_csv('./preprocessed_train_source_code.csv')
    test_source_code_df.to_csv('./preprocessed_test_source_code.csv')

if __name__ == "__main__":
    preprocessing()
import pandas as pd

def import_data(path = "/home/tegbe/2023/ccl/timeRobert/data/time-series-new.xlsx"):
    df = pd.read_excel(path)
    return df

def train_test_split_by_date(df, date_column, target_column, cutoff_date):
    """
    Split the data into training and testing sets based on a cutoff date.

    Parameters:
    - df: DataFrame containing the data
    - date_column: Name of the column containing dates
    - target_column: Name of the column containing the target variable
    - cutoff_date: Date to split the data for training and testing

    Returns:
    - train_data: DataFrame of training data
    - test_data: DataFrame of testing data
    """

    # Make a copy of the DataFrame
    df = df.copy()

    # Convert date_column to datetime
    df[date_column] = pd.to_datetime(df[date_column])

    # Split the data based on the cutoff date
    train_data = df[df[date_column] < cutoff_date]
    test_data = df[df[date_column] >= cutoff_date]

    # Extract X_train, y_train, X_test, y_test
    X_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]
    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]

    return X_train, y_train, X_test, y_test

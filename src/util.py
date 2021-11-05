import numpy as np
import pandas as pd

# list of columns that we want to use for out training and testing of our models
cols = ['Pclass','Sex','AgeGroup','SibSp','Parch','Embarked','Survived',
        'cabin_adv','cabin_multiple','numeric_ticket', 'name_title']

train_data_ratio = 0.8
valid_data_ratio = 0.1
test_data_ratio = 0.1

def create_age_groups(df):
    """
    Creates a new column on the given data frame by grouping age as per the following order:
        [ 0 -  2]  => Infant
        [ 2 -  4]  => Toddler
        [ 4 - 13]  => Kid
        [13 - 20]  => Teen
        [20 - --]  => Adult
        [  NAN  ]  => Unknown
    
    Args:
        df: pandas DataFrame containing all the data frame
    
    Returns:
        new pandas DataFrame with a new column ("AgeGroup") added to it
    """
    bins = [0,2,4,13,20,110]
    labels = ['Infant','Toddler','Kid','Teen','Adult']

    age_groups = pd.cut(df['Age'], bins=bins, labels=labels, right=False, retbins=True)[0]
    df['AgeGroup'] = age_groups
    df['AgeGroup'].replace(np.nan, 'Unknown', inplace=True)
    return df

def add_name_title(df):
    """
    Exracts name title from the name column and adds it as a new column ("NameTitle")
    
    Args:
        df: pandas DataFrame containing all the data frame
    
    Returns:
        new pandas DataFrame with a new column ("NameTitle") added to it
    """
    df['name_title'] = df.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())
    return df


def load_data(filepath):
    """
    Loads the titanic data from the provided filepath and does some pre-processing
    to prepare data for use in our model training/testing

    Args:
        filepath: str of file path to be loaded
    
    Returns:
        a list of tuples with data separated in train, valid, test where each tuple has x,y data,
        i.e. train_x, train_y = list[0]
             valid_x, valid_y = list[1]
             test_x, test_y = list[2]
    """
    df = pd.read_csv(filepath, index_col="PassengerId")

    # Add age group to our data
    df = create_age_groups(df)

    # extract name title
    df = add_name_title(df)

    #drop null 'embarked' rows. Only 2 instances of this in training and 0 in test 
    df.dropna(subset=['Embarked'], inplace = True)

    # create categorical variables
    df['cabin_multiple'] = df.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
    df['cabin_adv'] = df.Cabin.apply(lambda x: str(x)[0])
    df['numeric_ticket'] = df.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
    df.Pclass = df.Pclass.astype(str)

    # create dummy variables from categories
    all_dummies = pd.get_dummies(df[cols])

    # Randomize the dataset
    data_randomized = all_dummies.sample(frac=1, random_state=1)

    # Calculate index to split data into train/validate/test
    train_len = round(len(data_randomized) * train_data_ratio)
    valid_len = round(len(data_randomized) * valid_data_ratio)
    valid_end_idx = train_len + valid_len
    

    # Split into training and test sets
    train = data_randomized[:train_len]
    valid = data_randomized[train_len:valid_end_idx]
    test = data_randomized[valid_end_idx:]

    # extract label column for each set
    train_y = train.Survived
    train_x = train.drop(['Survived'], axis = 1)

    valid_y = valid.Survived
    valid_x = valid.drop(['Survived'], axis = 1)

    test_y = test.Survived
    test_x = test.drop(['Survived'], axis = 1)

    return [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]

if __name__ == '__main__':
    filename = "../data/train.csv"
    data = load_data(filename)
    train_x, train_y = data[0]
    print(train_x.shape)
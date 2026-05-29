from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split


def preprocess(df):
    y=df['label']
    X=df.drop(columns=['label'])
    cat=[c for c in X.columns if X[c].dtype=='object']
    num=[c for c in X.columns if c not in cat]
    pre=ColumnTransformer([
        ('num',MinMaxScaler(),num),
        ('cat',OneHotEncoder(handle_unknown='ignore'),cat)
    ])
    Xp=pre.fit_transform(X)
    return train_test_split(Xp,y,test_size=0.2,random_state=42),pre

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

class CustomPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.le_default = LabelEncoder()
        self.le_grade = LabelEncoder()
        self.intent_cols = None
        self.home_cols = None

    def fit(self, df, y=None):

        # Fit label encoders
        self.le_default.fit(df['cb_person_default_on_file'])
        self.le_grade.fit(df['loan_grade'])

        # Fit dummy columns
        self.intent_cols = pd.get_dummies(df['loan_intent']).columns.tolist()
        self.home_cols = pd.get_dummies(df['person_home_ownership']).columns.tolist()

        return self

    def transform(self, df):

        df = df.copy()

        # Label encoding
        df['cb_person_default_on_file'] = self.le_default.transform(df['cb_person_default_on_file'])
        df['loan_grade'] = self.le_grade.transform(df['loan_grade'])

        # Dummy encode
        intent_dummies = pd.get_dummies(df['loan_intent'])
        home_dummies = pd.get_dummies(df['person_home_ownership'])

        # Ensure ALL expected columns exist
        for col in self.intent_cols:
            if col not in intent_dummies:
                intent_dummies[col] = 0

        for col in self.home_cols:
            if col not in home_dummies:
                home_dummies[col] = 0

        # Align correct column order
        intent_dummies = intent_dummies[self.intent_cols]
        home_dummies = home_dummies[self.home_cols]

        df.drop(['loan_intent', 'person_home_ownership'], axis=1, inplace=True)

        df = pd.concat([df, intent_dummies, home_dummies], axis=1)

        return df

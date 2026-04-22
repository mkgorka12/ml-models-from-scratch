import pandas as pd


def calc_gini(df: pd.DataFrame, feature: str, target: str):
    feature_true_target_true = len(df[(df[feature] == True) & (df[target] == True)])
    feature_true_target_false = len(df[(df[feature] == True) & (df[target] == False)])

    feature_false_target_true = len(df[(df[feature] == False) & (df[target] == True)])
    feature_false_target_false = len(df[(df[feature] == False) & (df[target] == False)])

    feature_true = feature_true_target_true + feature_true_target_false
    feature_false = feature_false_target_true + feature_false_target_false

    prob_feature_true_target_true = (
        feature_true_target_true / feature_true if feature_true else 0
    )
    prob_feature_true_target_false = (
        feature_true_target_false / feature_true if feature_true else 0
    )

    prob_feature_false_target_true = (
        feature_false_target_true / feature_false if feature_false else 0
    )
    prob_feature_false_target_false = (
        feature_false_target_false / feature_false if feature_false else 0
    )

    gini_true = (
        1 - (prob_feature_true_target_true**2) - (prob_feature_true_target_false**2)
    )
    gini_false = (
        1 - (prob_feature_false_target_true**2) - (prob_feature_false_target_false**2)
    )

    proportions_feature_true = feature_true / len(df) if len(df) else 0
    proportions_feature_false = feature_false / len(df) if len(df) else 0

    gini = (gini_true * proportions_feature_true) + (
        gini_false * proportions_feature_false
    )
    return gini

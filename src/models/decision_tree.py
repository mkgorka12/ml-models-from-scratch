import pandas as pd
from abc import ABC, abstractmethod
from src.util.separation_metrics import calc_gini


class DecisionTreeElement(ABC):
    @abstractmethod
    def predict(self, record: pd.Series, verbose: bool = True):
        pass


class DecisionTreeLeaf(DecisionTreeElement):
    def __init__(self, value: bool):
        self.value = value

    def predict(self, record: pd.Series, verbose: bool = True):
        if verbose:
            print("Prediction =>", self.value)
        return self.value


class DecisionTreeNode(DecisionTreeElement):
    def __init__(self, feature: str):
        self.feature = feature
        self.true_child = None
        self.false_child = None

    def predict(self, record: pd.Series, verbose: bool = True) -> bool:
        pred = record.loc[self.feature]
        node = self.true_child if pred else self.false_child

        if verbose:
            print(self.feature + "?", pred)
        return node.predict(record, verbose=verbose)


class DecisionTreeBuilder:
    def __init__(self, target: str):
        self.target = target

    def _is_target_empty(self, df):
        target = df[self.target]
        return len(target) == 0

    def _is_target_same_value(self, df):
        target = df[self.target]
        return len(target) > 0 and (target.iloc[0] == target).all()

    def build(
        self, df: pd.DataFrame, parent_majority: bool = None
    ) -> DecisionTreeNode | DecisionTreeLeaf:
        if df.isnull().sum().sum():
            raise RuntimeError("This method doesn't allow any nans in the DataFrame")

        feature_names = df.drop(columns=self.target).columns

        # Check if we already had splitted target and it's empty now (end of recursion)
        if self._is_target_empty(df):
            return DecisionTreeLeaf(parent_majority)

        counts = df[self.target].value_counts()
        # Check if target is perfectly splitted in half
        # If there aren't going to be need to split decision further
        # then return parent's decision
        if len(counts) > 1 and counts.iloc[0] == counts.iloc[1]:
            current_majority = parent_majority
        else:
            current_majority = counts.idxmax()

        # Check if every record stores the same target value
        # or if there are any features left in the DataFrame (no point in further splitting)
        if self._is_target_same_value(df) or feature_names.empty:
            return DecisionTreeLeaf(current_majority)

        # Calculate GINI for each feature
        features_ginies = []
        for feature in feature_names:
            gini = calc_gini(df, feature, self.target)
            features_ginies.append((feature, gini))

        # Select the best separating feature
        feature_gini_sorted = sorted(features_ginies, key=lambda x: x[1])
        best_separating_feature = feature_gini_sorted.pop(0)[0]

        # Create parent node
        parent_node = DecisionTreeNode(best_separating_feature)

        df_best_separating_feature_true = df[df[best_separating_feature] == True].drop(
            columns=best_separating_feature
        )
        df_best_separating_feature_false = df[
            df[best_separating_feature] == False
        ].drop(columns=best_separating_feature)

        # Create leafs
        parent_node.true_child = self.build(
            df_best_separating_feature_true, current_majority
        )
        parent_node.false_child = self.build(
            df_best_separating_feature_false, current_majority
        )

        return parent_node

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from sklearn.utils.multiclass import unique_labels

from helper import ROOT
from helper.analysis import JSON_FOLDER, get_dataset_from_json

__all__ = [
    "convert_data",
    "train_extract_most_relevant",
    "plot_most_relevant",
    "save_tables",
]

RESULTS_FOLDER = Path(fr"{ROOT}/results/")
if not RESULTS_FOLDER.exists():
    RESULTS_FOLDER.mkdir()

TABLES_FOLDER = RESULTS_FOLDER / "tables"
if not TABLES_FOLDER.exists():
    TABLES_FOLDER.mkdir()

FIGS_FOLDER = RESULTS_FOLDER / "figs"
if not FIGS_FOLDER.exists():
    FIGS_FOLDER.mkdir()


def convert_data(file: Path) -> Dict[str, Any]:
    X_dict, y_str = get_dataset_from_json(file)
    dict_vectorizer = DictVectorizer(sparse=True)
    encoder = LabelEncoder()
    X, y = dict_vectorizer.fit_transform(X_dict), encoder.fit_transform(y_str)
    return {
        "X": X,
        "y": y,
        "encoder": encoder,
        "dict_vectorizer": dict_vectorizer,
    }


def train_extract_most_relevant(
    *,
    X: np.ndarray,
    y: np.ndarray,
    encoder: LabelEncoder,
    dict_vectorizer: DictVectorizer,
    k: int = 25,
    n: int = 10,
) -> Dict[str, Any]:

    chi2_selector = SelectKBest(chi2, k=k)
    X = chi2_selector.fit_transform(X, y)
    all_names = np.array(dict_vectorizer.get_feature_names())
    feature_names = list(all_names[chi2_selector.get_support()])

    scaler = StandardScaler(with_mean=False)
    X_, y_ = shuffle(X, y, random_state=24)

    X_ = scaler.fit_transform(X_)
    model = LogisticRegression()

    clf = model.fit(X_, y_)

    args = {"clf": clf, "feature_names": feature_names, "encoder": encoder, "n": n}

    return return_n_most_important(**args)


sns.set_style("whitegrid")


def plot_most_relevant(
    *, data: Dict[str, pd.DataFrame], translator: str, file: Path
) -> None:
    """Saves a bar plot of the most relevant features for a translator using a classifier.

    The function takes a list of data frames with the n most relevant weights and features
    for a classifier for a translator.

    Parameters:
    data: Dict[str, pd.DataFrame]  - The key is the translator name and the DataFrame
                                 contains two Series: 'Weight' and 'Feature'
    translator: str             - Name of the translator
    file: Path                  - Feature set used to train the model

    Returns:
    None
    """
    model = "Logistic Regression"
    plot = sns.barplot(
        x=data[translator]["Weight"], y=data[translator]["Feature"], palette="cividis",
    )
    features = " ".join(file.stem.split("_")[2:])
    plot.set(title=file.stem.replace("_", " "))
    fig = plot.get_figure()
    fig.savefig(
        FIGS_FOLDER / f"{file.stem}_{translator}.svg", bbox_inches="tight", dpi=300,
    )
    fig.clf()


def save_tables(*, df: pd.DataFrame, translator: str, file: Path) -> None:
    """Saves to disk the tabular data of the n most relevant features of a classifier.

    Takes a DataFrame containing the n most relevant features and their weights.

    Parameters:
    df: pd.DataFrame        - Contains two series: 'Weights' and 'Features'
    translator: str         - Name of the translator
    file: Path              - Feature set used to train the classifier

    Returns:
    None
    """
    latex = df.to_latex(float_format=lambda x: "%.4f" % x)
    with open(TABLES_FOLDER / f"{file.stem}_{translator}.tex", "w") as f:
        f.write(latex)


# def plot_pca(*, file: Path) -> None:
#     pca = PCA(n_components=2)
#     sns.set_style("whitegrid")
#     title = file.stem.replace("_", " ")

#     X_dict, y_str = get_dataset_from_json(file)

#     dict_vectorizer = DictVectorizer(sparse=False)
#     encoder = LabelEncoder()

#     X, y = dict_vectorizer.fit_transform(X_dict), encoder.fit_transform(y_str)

#     features = StandardScaler().fit_transform(X)
#     X_pca = pca.fit_transform(features)

#     d = {
#         "Principal Component 1": pd.Series(X_pca[:, 0]),
#         "Principal Component 2": pd.Series(X_pca[:, 1]),
#         "Translator": pd.Series(encoder.inverse_transform(y)),
#     }
#     data = pd.DataFrame(d)

#     plot = sns.scatterplot(
#         x="Principal Component 1",
#         y="Principal Component 2",
#         hue="Translator",
#         data=data,
#         palette="cividis",
#         alpha=0.75,
#     )

#     plot.set(title=f"{title}")
#     fig = plot.get_figure()
#     fig.savefig(
#         PCA_FOLDER / f"pca_{'_'.join(title.split())}.svg", bbox_inches="tight", dpi=300,
#     )
#     fig.clf()

#     return None


def return_n_most_important(
    *,
    clf: LogisticRegression,
    feature_names: List[str],
    encoder: LabelEncoder,
    n: int = 10,
) -> Dict[str, DataFrame]:
    """Returns n features with largest weights in a logistic regression classifier.

    As inputs a trained logistic regressor along with the DictVectorizer and
    LabelEncoder used to encode dictionary of feature names and values and
    classes names, and the number of features to return.

    Parameters:
    clf:            LogisticRegression    - A trained logistic regression classifier
    feature_names:  List[str]             - List of strings with the features name used in clf
                                            dict of features and counts to a numpy array
    encoder:        LabelEncoder          - Scikit-learn LabelEncoder used to encoding classes names  
    n:              int                   - number of most relevant features used

    Returns:
    A dictionary that maps name of class to DataFrame of n most relevant features and
    their weights.
    """
    most_important: defaultdict = defaultdict(DataFrame)
    classes_names = encoder.classes_
    # feature_names = v.get_feature_names()
    columns = ["Feature", "Weight"]
    if len(classes_names) == 2:

        indices = clf.coef_[0].argsort()[-n:][::-1]
        data = []
        for index in indices:
            data.append([feature_names[index], clf.coef_[0][index]])
        class_name = encoder.inverse_transform([1])[0]
        most_important[class_name] = DataFrame(
            data, columns=columns, index=range(1, n + 1)
        )

        indices = (-clf.coef_[0]).argsort()[-n:][::-1]
        data = []
        for index in indices:
            data.append([feature_names[index], (-clf.coef_[0])[index]])
        class_name = encoder.inverse_transform([0])[0]
        most_important[class_name] = DataFrame(
            data, columns=columns, index=range(1, n + 1)
        )

    elif len(classes_names) > 2:

        for i in range(len(classes_names)):
            indices = clf.coef_[i].argsort()[-n:][::-1]  # n largest elements
            data = []
            for index in indices:
                data.append([feature_names[index], clf.coef_[i][index]])
            class_name = encoder.inverse_transform([i])[0]
            most_important[class_name] = DataFrame(
                data, columns=columns, index=range(1, n + 1)
            )
    return dict(most_important)

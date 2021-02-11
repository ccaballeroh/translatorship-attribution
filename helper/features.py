from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
from helper.utils import return_n_most_important

__all__ = [
    "convert_data",
    "train_extract_most_relevant",
    "plot_most_relevant",
    "save_tables",
    "plot_confusion_matrix",
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

CONF_MAT_FOLDER = FIGS_FOLDER / "cm"
if not CONF_MAT_FOLDER.exists():
    CONF_MAT_FOLDER.mkdir()

MOST_RELEVANT_FOLDER = FIGS_FOLDER / "most"
if not MOST_RELEVANT_FOLDER.exists():
    MOST_RELEVANT_FOLDER.mkdir()


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
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    encoder: LabelEncoder,
    dict_vectorizer: DictVectorizer,
    feature_selection: bool,
    k: int = 45,
    n: int = 15,
) -> Dict[str, Any]:

    if feature_selection and X.shape[1] >= k:
        chi2_selector = SelectKBest(chi2, k=k)
        X = chi2_selector.fit_transform(X, y)
        all_names = np.array(dict_vectorizer.get_feature_names())
        feature_names = list(all_names[chi2_selector.get_support()])
    else:
        feature_names = dict_vectorizer.get_feature_names()

    scaler = StandardScaler(with_mean=False)
    X_, y_ = shuffle(X, y, random_state=24)

    if model_name == "LogisticRegression":
        X = scaler.fit_transform(X)
        model = LogisticRegression()
    elif model_name == "SVM":
        X = scaler.fit_transform(X)
        model = LinearSVC()
    elif model_name == "NaiveBayes":
        model = MultinomialNB()
    else:
        raise NotImplementedError

    clf = model.fit(X_, y_)

    args = {"clf": clf, "feature_names": feature_names, "encoder": encoder, "n": n}

    most_relevant = return_n_most_important(**args)

    return {
        "clf": clf,
        "scaler": scaler,
        "most_relevant": most_relevant,
    }


sns.set_style("whitegrid")


def plot_most_relevant(
    *, data: Dict[str, pd.DataFrame], translator: str, model: str, file: Path
) -> None:
    """Saves a bar plot of the most relevant features for a translator using a classifier.

    The function takes a list of data frames with the n most relevant weights and features
    for a classifier for a translator.

    Parameters:
    data: Dict[str, pd.DataFrame]  - The key is the translator name and the DataFrame
                                 contains two Series: 'Weight' and 'Feature'
    translator: str             - Name of the translator
    model: str                  - Name of the model (classifier) used
    file: Path                  - Feature set used to train the model

    Returns:
    None
    """
    plot = sns.barplot(
        x=data[translator]["Weight"], y=data[translator]["Feature"], palette="cividis",
    )
    features = " ".join(file.stem.split("_")[2:])
    plot.set(title=f"{translator} - {model} - {features}")
    fig = plot.get_figure()
    fig.savefig(
        MOST_RELEVANT_FOLDER / f"{file.stem}_{translator}_{model}.png",
        bbox_inches="tight",
        dpi=300,
    )
    fig.clf()


def save_tables(
    *, df: pd.DataFrame, translator: str, file: Path, model_name: str
) -> None:
    """Saves to disk the tabular data of the n most relevant features of a classifier.

    Takes a DataFrame containing the n most relevant features and their weights.

    Parameters:
    df: pd.DataFrame        - Contains two series: 'Weights' and 'Features'
    translator: str         - Name of the translator
    file: Path              - Feature set used to train the classifier
    model_name: str         - Name of the classifier used

    Returns:
    None
    """
    df.to_csv(
        TABLES_FOLDER / f"{file.stem}_{translator}_{model_name}.csv",
        float_format="%.4f",
    )

    latex = df.to_latex(float_format=lambda x: "%.4f" % x)
    with open(TABLES_FOLDER / f"{file.stem}_{translator}_{model_name}.tex", "w") as f:
        f.write(latex)

    html = df.to_html(float_format="%.4f")
    with open(TABLES_FOLDER / f"{file.stem}_{translator}_{model_name}.html", "w") as f:
        f.write(html)


def plot_confusion_matrix(
    X: np.ndarray, y: np.ndarray, encoder: LabelEncoder, file: Path
) -> None:
    sns.set(font_scale=1.4)
    scaler = StandardScaler(with_mean=False)
    X = scaler.fit_transform(X)
    X_, y_ = shuffle(X, y, random_state=24)
    log_model = LogisticRegression()

    y_pred = cross_val_predict(log_model, X_, y_, cv=10)
    cm = confusion_matrix(y_, y_pred, labels=unique_labels(y_))
    df = pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_)
    cm_plot = sns.heatmap(
        df, annot=True, cbar=None, cmap="Blues", fmt="d", annot_kws={"size": 18}
    )
    plt.title(f"{' '.join(file.stem.split('_')[2:])}")
    plt.tight_layout()
    plt.ylabel("True translator")
    plt.xlabel("Predicted translator")
    plt.savefig(
        CONF_MAT_FOLDER / f"cm_{file.stem}.png", bbox_inches="tight", dpi=300,
    )
    plt.clf()

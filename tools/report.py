from .util import labeled_progress

import sklearn, json, itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import ipywidgets as widgets
from IPython.display import display

from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from collections import OrderedDict


import warnings

# Dimensionality reduction
from sklearn.decomposition import PCA

# Model evaluation
from sklearn.metrics import confusion_matrix, f1_score

# Optimization mechanics
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Classifiers
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

classifiers = {
    "K-Best" : [{
        "classify": [KNeighborsClassifier()],
        'classify__n_neighbors': [3,5,8,13,21]
        }],
    "SVC" : [{
        "classify": [SVC()],
        'classify__C': [.025, .1, .25, .5, 1],
        "classify__kernel": ["rbf", "linear"], 
        "classify__gamma": ['auto', 2]
    }],
    "Gausian Process": [{
        "classify": [GaussianProcessClassifier()],
        "classify__kernel": [None],
        "classify__optimizer": ['fmin_l_bfgs_b', None],
        "classify__multi_class": ["one_vs_rest", "one_vs_one"]
    }],
    "Decision Tree": [{
        "classify": [DecisionTreeClassifier()],
        "classify__criterion": ["gini", "entropy"],
        "classify__splitter": ["best", "random"],
        "classify__max_depth": [2,3,5,8,13,21]
    }], 
    "Random Forest": [{
        "classify" : [RandomForestClassifier()],
        "classify__max_depth" : [2,3,5,8,13],
        "classify__n_estimators" : [1,10,25, 50],
        "classify__max_features" : [1,2,3]
    }], 
    "Multi-layer Perceptron": [{
        "classify" : [MLPClassifier()],
        "classify__alpha": [.1,.5, 1, 2, 3]
    }],
    "Adaptive Boost": [{
        "classify" : [AdaBoostClassifier()],  #DecisionTreeClassifier is default base classifier
        "classify__n_estimators": [10, 25, 50, 100],
        "classify__algorithm" : ["SAMME.R", "SAMME"]
    }]
}

def first(iterable): return next(iter(iterable))

def _mat(y, pred, normalize, classes):
    cm = confusion_matrix(y, pred, labels=classes)
    if normalize: cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return cm


def _corange(m1, m2, *, normalize=False):
    if normalize: return 0, 1

    mx = max(m1.max(), m2.max())
    mn = min(m1.min(), m2.min())
    return mn, mx


def score_sort(result, X, y):
    """
    Sort a set of conditions by how they score on a specific dataset.
    result -- Thigns to sort
    X, y -- Test data/ground-truth labels
    """

    ordered = sorted(result.items(), key=lambda e: e[1].score(X, y), reverse=True)
    return OrderedDict(ordered)

def train(X, y, pipeline, *, grid_opts={"n_jobs": -1}, **grids):
    """
    Train a classifiers over the given grids.  
    Return values are keyed by the key in the grid name
    
    X -- Dataframe of observations to train on
    y -- Series of labels of labels to train on
    pipeline -- pipeline to use
    grid -- grid to search
    grid_opts -- parameters to pass to the grid search on construction
    Returns: {id: trained-classifier}
    """
            
    def _search(pipeline, grid):
        clf = GridSearchCV(pipeline, grid, **grid_opts)
        clf.fit(X, y)
        return clf
    
    if len(grids) == 0: 
        pipeline.fit(X,y)
        return pipeline

    s = labeled_progress(grids.items(), len(grids), grids.keys())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = {key:_search(pipeline, grid) for key, grid in s}

    return results


def reduction_impact(pipeline, X, y, *, grid_opts={"n_jobs": -1}, **grids):
    """
    Train classifiers with and without the proided reducers.  
    
    reduce -- reducer to use
    grid -- grid to search
    X -- Dataframe of observations to train on
    y -- Series of labels of labels to train on
    grid_opts -- parameters to pass to the grid search on construction
    Returns: {id: w/reduction} {id: w/o reuduction)
    """
        
    control_report = widgets.IntProgress(value=0,min=0, max=len(grids),
        description='Control', bar_style='info', orientation='Horizontal'
    )
    reduced_report = widgets.IntProgress(value=0,min=0, max=len(grids),
        description='Reduced', bar_style='', orientation='Horizontal'
    )
    
    detail = widgets.HTML(value='<i>initializing</i>', disabled=True)
    display(widgets.VBox([control_report, reduced_report, detail]))
    
    def _search_each(pipeline, grid, progress):
        return {name: _search(pipeline, config, name, progress) 
                for name, config in grid.items()}
        
    def _search(pipeline, grid, label, progress):
        detail.value = "Searching " + label

        clf = GridSearchCV(pipeline, grid, **grid_opts)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X, y)
            
        progress.value = progress.value+1
        return clf
    
    p2 = Pipeline(steps=[e for e in pipeline.steps if e[0] != "reduce_dim"])
    refs = _search_each(p2, grids, control_report)

    ress  = _search_each(pipeline, grids, reduced_report)

    detail.value = "Search complete"

    return ress, refs


def overall_scores(reduction, no_reduction, X, y, *, cmap_at=200, cmaps=[plt.cm.Purples, plt.cm.Greens], ax=None):
    """
    Plots a graph of scores for each of the condtiions. 
    Assumes all keys in `reduction` are also in `no_reduction`.
    """
    
    scores = [(title, reduction[title].score(X, y), 
                       no_reduction[title].score(X, y)) 
                 for title in reduction.keys()]
    
    cmap = [cmap(cmap_at) for cmap in cmaps]
    
    df = pd.DataFrame(data=scores,
                        columns=["Classifier", "With Reduction", "No Reduction"])\
                .set_index("Classifier")
        
    df.plot.bar(color=cmap, ax=ax)

def detail_on_label(left_pred, right_pred, y, *, 
           left_label="With Reduction", right_label="No Reduction",
           classes=None, normalize=False,
           title="", 
           axs=None):
    """
    Plot a detailed comparison of two sets of predictions (left & right).
    left_pred, right_pred -- The prediction sets
    y -- The ground-truth labels
    left_label, right_label -- (optional) Label for the left/red predictions 
    classes -- (optional) List of expected classes.  Also controls the display order.
    normalize -- (optional) Should the confusion matrix be normalized or show count values?
    title -- (optional) Overall plot title
    axs -- (optional) Axes to put the plots in.  Needs three.
    """

    def _abrv(label): 
        root = label.split(" ")[0]
        return root if len(root) <= 4 else root[:3] + "."
    
    def _errplot(focus, context, cmap, title, ax, *, normalize=False, cm_labels=None):
        title = title + (" (normalized)" if normalize else "")
        
        mask = np.empty(focus.shape, dtype=np.bool)
        mask.fill(True)
        np.fill_diagonal(mask, [False])

        vmin, vmax = _corange(focus[mask], context[mask], normalize=normalize)
        ax = sns.heatmap(focus, mask=~mask, cmap=cmap, annot=True,
                    cbar_kws={"orientation": "horizontal"},
                   ax=ax, fmt='.2f', square=True,
                   xticklabels=classes, yticklabels=classes,
                        vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Truth")
                
        return ax
    
    def _accplot(left, right, refcmap, rescmap, *, ax=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            left = f1_score(y, left, average=None)*-1
            right = f1_score(y, right, average=None)
            
        scores = np.vstack([left, right])

        left_cmap = np.flip(refcmap(np.arange(0,1,.1)), axis=0)
        right_cmap = rescmap(np.arange(0,1,.1))
        joint = np.append(left_cmap, right_cmap,  axis=0)
        cmap = LinearSegmentedColormap.from_list("REF-RES", joint)

        ax = sns.heatmap(scores.T, cmap=cmap, annot=True,
                        ax=ax, center=0, fmt='.2f', square=True,
                        xticklabels=[_abrv(left_label), _abrv(right_label)], yticklabels=classes,
                        vmin=-1, vmax=1,
                        cbar=False)
        
        ax.set_title("F1 Accuracy")
        for t in ax.texts: t.set_text(t.get_text().replace("-", ""))

        ax.set_ylabel(title, weight="bold", fontsize=20)
        return ax

    if classes is None:
        classes = set(left_pred).union(right_pred).union(y)
        classes = sorted(classes)
        
    left_mat = _mat(y, left_pred, normalize, classes)
    right_mat = _mat(y, right_pred, normalize, classes)
    
    if axs is None:
        fig = plt.figure(figsize=(15,5))
        axs = [plt.subplot(1,3,1), plt.subplot(1,3,2), plt.subplot(1,3,3)]
    
    acc = _accplot(left_pred, right_pred, plt.cm.Purples, plt.cm.Greens, ax=axs[0])
    err_res = _errplot(left_mat, right_mat, plt.cm.Purples, 
                       f"Errors -- {left_label}", 
                       ax=axs[1], normalize=normalize)
    err_ref = _errplot(right_mat, left_mat, plt.cm.Greens, 
                       f"Errors -- {right_label}", 
                       ax=axs[2], normalize=normalize)

    return acc, err_res, err_ref

def compare(X, y, *args, normalize=True, **kwargs):
    """
    Comre two classifier from the condtiions.
    X, y -- Test data an ground truth.
    
    *args as (name, name, conditions)
    *args as (name, conditions)
    **kwargs as (name=clf, name=clf) -- Compare the
    **kwargs as (name=clf)
    
    """

    if len(args) == 3:
        left_label, right_label, conditions = args
        title = f"{left_label} vs. {right_label}"
        left = conditions[left_label]
        right = conditions[right_label]
    elif len(args) == 1 and len(kwargs) ==2:
        title = args[0]
        left_label, right_label = [k.replace("_", " ") for k in kwargs.keys()]
        left_conditions, right_conditions = kwargs.values()
        
        left = left_conditions[title]
        right = right_conditions[title]
    elif len(kwargs) == 2:
        left_label, right_label = [k.replace("_", " ") for k in kwargs.keys()]
        title = f"{left_label} vs. {right_label}"
        left, right = kwargs.values()
    else:
        raise ValueError("""Must specify conditions in arguments:  
                              (1) label,label,conditions, 
                              (2) label, conditions, conditions or 
                              (3) two kwargs)""")
    
    left_pred = left.predict(X)
    right_pred = right.predict(X)
    
    detail_on_label(left_pred, right_pred, y, title=title, 
                    left_label=left_label, right_label=right_label, 
                    normalize=normalize)

    
    df = pd.DataFrame(data={left_label: left_pred, right_label: right_pred, "Truth": y}, 
                      columns=["Truth", left_label, right_label])
    return df
    
def summary(X, y, reduction, no_reduction, *, delta=False, classes=None):
    """
    Provide a summary of all conditions and detail report on "Best" condition.
    """
    reduction = score_sort(reduction, X, y)
    
    title, res = first(reduction.items())
    ref = no_reduction[title]
        
    ref_pred = ref.predict(X)
    res_pred = res.predict(X)
    #res_pred = np.array(["high"]* ref_pred.shape[0])

    fig = plt.figure(figsize=(15,10))
    f1 = plt.subplot2grid((3, 5), (0,0), colspan=1, rowspan=2)
    e1 = plt.subplot2grid((3, 5), (0,1), colspan=2, rowspan=2)
    e2 = plt.subplot2grid((3, 5), (0,3), colspan=2, rowspan=2)
    summary = plt.subplot2grid((3, 5), (2,0), colspan=5)
    
    rslt = detail_on_label(res_pred, ref_pred, y, 
                  classes=classes,
                  normalize=True, 
                  title=title, 
                axs=[f1,e1,e2])
    
    overall_scores(reduction, no_reduction, X, y, cmap_at=175, ax=summary)
    

def detail(X, y, clf, label="", normalize=True, classes=None):
    def _accplot(pred, refcmap, classes, title, *, ax=None, normalize=False):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = f1_score(y, pred, average=None)

        range = refcmap(np.arange(0,1,.1))
        cmap = LinearSegmentedColormap.from_list("REF-RES", range)
        ax = sns.heatmap(np.array([scores]).T, cmap=cmap, annot=True,
                        ax=ax, center=0, fmt='.2f', square=True,
                        xticklabels=[label], yticklabels=classes,
                        vmin=-1, vmax=1, 
                        cbar=False)
        
        ax.set_title("Accuracy (f1)")
        ax.set_ylabel(title, weight="bold", fontsize=20)

        
    def _errplot(focus, cmap, classes, title, *, ax, normalize=False):
        title = title + (" (normalized)" if normalize else "")

        mask = np.empty(focus.shape, dtype=np.bool)
        mask.fill(True)
        np.fill_diagonal(mask, [False])

        vmin, vmax = _corange(focus[mask], focus[mask], normalize=normalize)
        ax = sns.heatmap(focus, mask=~mask, cmap=cmap, annot=True,
                    cbar_kws={"orientation": "horizontal"},
                   ax=ax, fmt='.2f', square=True,
                   xticklabels=classes, yticklabels=classes,
                   vmin=vmin, vmax=vmax)
        
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Truth")

        return ax
    
    pred = clf.predict(X)

    if classes is None:
        classes = set(pred).union(y)
        classes = sorted(classes)
    
    matrix = _mat(y, pred, normalize, classes)
        
    df = pd.DataFrame(data={label: pred, "Truth": y}, 
                      columns=["Truth", label])
    
    fig = plt.figure(figsize=(8,5))
    acc_ax = plt.subplot2grid((1, 3), (0,0), colspan=1, rowspan=1)
    err_ax = plt.subplot2grid((1, 3), (0,1), colspan=2, rowspan=1)    

    acc = _accplot(pred, plt.cm.Purples, classes, label, ax=acc_ax, normalize=normalize)
    err = _errplot(matrix, plt.cm.Purples, 
                       classes, f"Errors -- {label}", 
                       ax=err_ax, normalize=normalize)
    
    return df

def config(source, classifier=None, *, X=None, y=None):
    """Return the best paramaeters for classifier.
    source -- trained classifier or dict of classifiers
    classifier -- String naming element from conditions to report parametsr for
    X/y -- If no classifier is supplied, used to sort conditions and returns parameters for highest scoring
    """
    
    try:
        return source.best_params_
    except: pass # swallow error
    
    if len(source) == 1: 
        return first(source.values()).best_params_
    if classifier is None:
        sorted = score_sort(source, X, y).values()
        return next(iter(sorted)).best_params_
    else:
        return source[classifier].best_params_


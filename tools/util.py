import json
import pandas as pd

from sklearn.base import TransformerMixin
from itertools import zip_longest

from ipywidgets import FloatProgress, HTML, HBox
from IPython.display import display

def iter_progress(it, n):
    
    f = FloatProgress(min=0, max=n)
    display(f)
    
    for x in it:
        yield x
        f.value += 1
        f.description = f'{int(100*f.value/n)}%'

def labeled_progress(it, n, labels, fillvalue="...", final=""):
    "Iterator and set of labels.  Reports progress with bar and label."
    
    detail = HTML(value='<i>initializing</i>', disabled=True)
    f = FloatProgress(min=0, max=n)
    
    
    display(HBox([f, detail]))
    
    for x, label in zip_longest(it, labels, fillvalue=fillvalue):
        detail.value = label
        yield x
        f.value += 1
        f.description = f'{int(100*f.value/n)}%'
        
    detail.value = final

        
class Checkpoint(TransformerMixin):
    def transform(self, X, **kwargs):
        self.X_transform = X
        
        return X

    def fit(self, X, y, **kwargs):
        self.X_fit = X
        self.y_fit = y
        
        return self
    
    def fit_transform(self, X, y, **kwargs):
        return self.fit(X, y).transform(X)


def load_dataset(name):
    with open(name) as fp:
        obj = json.load(fp)
        X = pd.DataFrame(**obj['X'])
        y = pd.Series(obj['y'], index=X.index)
        return X, y
""" Evalutator class that loads metric using method path and kwargs from a config dictionary
"""
from pathlib import Path
import json
import inspect
from ssl_graph_encoder.utils.model_io import load_class
from sklearn.preprocessing import StandardScaler

class EmbeddingMetric:
    def __init__(self, name, method, kwargs):
        self.name = name
        self.method = method
        self.kwargs = kwargs
    
    def evaluate_unsupervised(self, embeddings):
        return self.method(embeddings, **self.kwargs)
    
    def evaluate_supervised(self, embeddings, labels, split_idxs):
        return self.method(embeddings, labels, split_idxs, **self.kwargs)

class EmbeddingEvaluator:
    """ Upon initialisation, load metric methods
        When called, evaluate metrics
    """
    def __init__(self, metric_config):
        self.unsupervised_metrics = []
        self.supervised_metrics = []
        for key, val in metric_config.items():
            method = load_class("ssl_graph_encoder.utils.metrics." + key) # works for methods too
            kwargs = val.get("kwargs", {}) # if kwargs exist, get them
            
            args = inspect.getfullargspec(method).args
            if "labels" in args:
                self.supervised_metrics.append(EmbeddingMetric(key, method, kwargs))
            else:
                self.unsupervised_metrics.append(EmbeddingMetric(key, method, kwargs))

        # for storing results
        self.results = {}

    def __call__(self, embeddings, labels, outfile):
        self.unsupervised_eval(embeddings)
        self.supervised_eval(embeddings, labels)
        self.flush_results(outfile)

    def unsupervised_eval(self, embeddings):
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)
        for m in self.unsupervised_metrics:
            res = m.evaluate_unsupervised(embeddings)
            if isinstance(res, dict):
                self.results.update(res)
            elif isinstance(res, list):
                for count, r in enumerate(res):
                    self.results[m.name + f"_{count}"] = r
            else:
                self.results[m.name] = res
    
    def supervised_eval(self, embeddings, labels, split_idxs):
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)
        for m in self.supervised_metrics:
            res = m.evaluate_supervised(embeddings, labels, split_idxs)
            if isinstance(res, dict):
                self.results.update(res)
            elif isinstance(res, list):
                for count, r in enumerate(res):
                    self.results[m.name + f"_{count}"] = r
            else:
                self.results[m.name] = res

    def flush_results(self, outfile):
        with open(Path(outfile), "a", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)
        self.results = {}


import json
import numpy as np

from readers import CDReader

query_funcs = []


def preprocess(dirpath, query_results):
    """Iterates over a dataset and sends each data point
    to each coroutine decorated by query function
    """
    reader = CDReader(dirpath, batched=True, batch_size=20)
    coroutines = [cr(query_results) for cr in query_funcs]
    for cr in coroutines:
        next(cr)
    for X, Y in reader.read():
        for i in range(len(Y)):
            for cr in coroutines:
                cr.send((X[i], Y[i]))
    for cr in coroutines:
        cr.close()


def query(func):
    query_funcs.append(func)
    return func


@query
def size(query_results):
    cat_freq = {}
    total = 0
    try:
        while True:
            img, cat = (yield)
            cat_freq[cat] = cat_freq.get(cat, 0) + 1
            total += 1
    except GeneratorExit:
        query_results['total'] = total
        query_results['cat_freq'] = cat_freq


@query
def avg_brightness(query_results):
    sum_of_averages = 0
    total = 0
    try:
        while True:
            img, cat = (yield)
            sum_of_averages += np.average(img)
            total += 1
    except GeneratorExit:
        query_results['averge_brightness'] = sum_of_averages/total


class QueryExecutor:
    """The query execution engine

    Usage:
    ------
    >> qe = QueryExecutor(<dataset_path>)
    >> # run queries now
    >> print(qe.size())
    >> print(qe.nb_categories())
    """

    cached_queries_path = 'cached_queries.json'
    cached_queries = None

    @classmethod
    def _load_queries(cls):
        with open(cls.cached_queries_path, 'r') as fp:
            return json.load(fp)

    @classmethod
    def _save_queries(cls):
        with open(cls.cached_queries_path, 'w') as fp:
            json.dump(cls.cached_queries, fp)

    def __init__(self, dirpath):
        self.dirpath = dirpath
        if QueryExecutor.cached_queries is None:
            QueryExecutor.cached_queries = QueryExecutor._load_queries()
        if dirpath not in QueryExecutor.cached_queries:
            QueryExecutor.cached_queries[dirpath] = results = {}
            preprocess(dirpath, results)
            QueryExecutor._save_queries()

    def size(self, category=None):
        results = QueryExecutor.cached_queries[self.dirpath]
        if category:
            return results['cat_freq'].get(category, 0)
        else:
            return results['total']

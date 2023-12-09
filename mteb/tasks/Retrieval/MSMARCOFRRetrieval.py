from collections import defaultdict
from datasets import load_dataset, DatasetDict

from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval


def load_retrieval_data(hf_hub_name, eval_splits):
    eval_split = eval_splits[0]
    dataset = load_dataset(hf_hub_name, 'data')
    qrels = load_dataset(hf_hub_name, 'qrels', split="qrels")

    corpus = {e['id']: {'text': e['text']} for e in dataset['corpus']}
    queries = {e['id']: e['text'] for e in dataset['queries']}
    relevant_docs = defaultdict(dict)
    for e in qrels:
        relevant_docs[e['query_id']][e['doc_id']] = e['relevance']

    corpus = DatasetDict({eval_split:corpus})
    queries = DatasetDict({eval_split:queries})
    relevant_docs = DatasetDict({eval_split:relevant_docs})
    return corpus, queries, relevant_docs

class MMarcoFRRetrieval(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            'name': 'MSMarcoFRRetrieval',
            'hf_hub_name': 'OrdalieTech/MMarcoRetrieval-FR-benchmark',
            'reference': 'https://github.com/unicamp-dl/mMARCO',
            'description': 'mMARCO is a multilingual version of the MS MARCO passage ranking dataset',
            'type': 'Retrieval',
            'category': 's2p',
            'eval_splits': ['test'],
            'eval_langs': ['fr'],
            'main_score': 'ndcg_at_10',
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(self.description['hf_hub_name'],
                                                                            self.description['eval_splits'])
        self.data_loaded = True


from ...abstasks.AbsTaskSTS import AbsTaskSTS

class OrdalieFRSTS(AbsTaskSTS):
    @property
    def description(self):
        return {
            "name": "OrdalieFRSTS",
            "hf_hub_name": "OrdalieTech/Ordalie-FR-STS-benchmark",
            "type": "STS",
            "category": "s2s",
            "eval_splits": ["test[:500]"],
            "eval_langs": ["fr"],
            "main_score": "cosine_spearman",
            "min_score": 0,
            "max_score": 1,
        }


import factory
from factory import fuzzy
from datetime import datetime, timedelta
import random

from models import Job, JobStatus, JobPriority

class JobFactory(factory.Factory):
    class Meta:
        model = Job

    job_type = fuzzy.FuzzyChoice(["train_sklearn_model"])
    user_id = factory.Sequence(lambda n: f"user_{n}")

    config = factory.LazyAttribute(lambda obj: {
        "model": "RandomForest",
        "n_estimators": random.randint(50, 500),
        "dataset_rows": random.randint(1000, 50000),
        "n_features": random.randint(10, 50)
    })

    priority = fuzzy.FuzzyChoice([p.value for p in JobPriority])
    status = fuzzy.FuzzyChoice([s for s in JobStatus])

    predicted_cpu_percent = fuzzy.FuzzyFloat(10.0, 80.0)
    predicted_memory_db = fuzzy.FuzzyFloat(100.0, 2000.0)

    created_at = factory.LazyFunction(datetime.now())
    
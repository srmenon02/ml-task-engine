from locust import HttpUser, task, between
import random

class TaskEngineUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        self.headers = {
            "Authorization": "Bearer test_api_key_loadtest"
        }

    @task(10)
    def create_job(self):
        self.client.post(
            "/jobs",
            json={
                "job_type": "train_sklearn_model",
                "config": {
                    "model": "RandomForest",
                    "n_estimators": random.randint(50, 200),
                    "dataset_rows": random.randint(1000, 10000)
                },
                "priority": random.randint(0, 10)
            },
            headers=self.headers
        )

    @task(5)
    def list_jobs(self):
        self.client.get("/jobs", headers = self.headers)

    @task(2)
    def get_system_stats(self):
        self.client.get("/system/stats", headers = self.headers)
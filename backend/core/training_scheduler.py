import structlog
from models import local_session, ResourceProfile
from core.predictor import get_predictor

logger = structlog.get_logger()

class TrainingScheduler:
    def __init__(self, retrain_interval: int = 10):
        self.retrain_interval = retrain_interval
        self.last_training_count = 0

    def check_and_retrain(self):
        db = local_session()

        try:
            profile_count = db.query(ResourceProfile).count()

            new_profiles = profile_count - self.last_training_count

            if new_profiles >= self.retrain_interval:
                logger.info(
                    "training scheduler retraining",
                    new_profiles = new_profiles,
                    total_profiles = profile_count
                )

                predictor = get_predictor()
                success = predictor.train(min_samples = 5)

                if success:
                    self.last_training_count = profile_count
                    logger.info("Scheduler retraining successful")
                else:
                    logger.warning("Scheduler retraining unsuccessful")


        except Exception as e:
            logger.error("training scheduler error", str(e))
        finally:
            db.close()

_scheduler = None
def get_training_scheduler() -> TrainingScheduler:
    global _scheduler
    return _scheduler if _scheduler else TrainingScheduler(retrain_interval = 10)

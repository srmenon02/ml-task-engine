from models import init_db
import structlog
import os
logger = structlog.get_logger()

if __name__ == "__main__":
    env = os.getenv("ENVIRONMENT", "development")
    logger.info(f"Creating Database Tables for {env} environment")
    init_db()
    logger.info("DB Initialization Successful")
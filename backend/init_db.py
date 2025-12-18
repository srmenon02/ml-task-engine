from models import init_db
import structlog

logger = structlog.get_logger()

if __name__ == "__main__":
    logger.info("Creating Database Tables")
    init_db()
    logger.info("DB Initialization Successful")
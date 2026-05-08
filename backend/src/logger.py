import logging

# Configure basic logging for the project
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class PipelineLogger:
    def __init__(self, enabled: bool = True, name: str = "Pipeline"):
        self.enabled = enabled
        self.logger = logging.getLogger(name)

    def info(self, msg: str):
        if self.enabled:
            self.logger.info(msg)

    def error(self, msg: str):
        if self.enabled:
            self.logger.error(msg)


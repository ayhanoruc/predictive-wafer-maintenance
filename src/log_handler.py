import logging 
import os 
from datetime import datetime 




class AppLogger():
    """
    A class for handling logging operations with dynamic log file paths based on phases.
    
    Attributes:
        phase_name (str): The name of the current ml cycle phase name.
        logs_path (str): The base directory for log files.
        logs_phase_path (str): The directory path for phase-specific log files.
        logger: The logger object for this phase.
    """

    def __init__(self, phase_name):
        """
        Initialize the AppLogger instance.
        
        Args:
            phase_name (str): The name of the ML cycle phase.
        """

        self.phase_name = phase_name
        self.logs_path = os.path.join(os.getcwd(), "logs")
        os.makedirs(self.logs_path, exist_ok=True)

        self.logs_phase_path = os.path.join(self.logs_path, self.phase_name)
        os.makedirs(self.logs_phase_path, exist_ok=True)


        self.logger = self._configure_logger()

    def _configure_logger(self):

        """
        Configure and return a logger instance in hourly basis.
        
        Returns:
            logging.Logger: A configured logger instance.
        """


        
        logger = logging.getLogger(self.phase_name) # logger initalization
        logger.setLevel(logging.INFO) # common use

        formatter = logging.Formatter("[%(asctime)s - %(name)s -- %(levelname)s -- %(message)s]")

        current_time = datetime.now()
        LOG_FILE_NAME = f"{current_time.strftime('%d-%m-%Y_%H')}.log"
        log_file_path = os.path.join(self.logs_phase_path, LOG_FILE_NAME)

        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def handle_logging(self, message: str,level=logging.INFO):

        """
        Log a message with the current timestamp into the phase-specific log file.
        
        Args:
            message (str): The log message to be recorded.
            level(logging.level): The severity of logging message
        """
        self.logger.log(level=level, msg=message)




# Example usage
if __name__ == "__main__":
    logger2 = AppLogger("data_validation")
    print(logger2.logs_phase_path)
    logger2.handle_logging("This is a test log message for data validation.")

    # Append more log messages to the same log file
    logger2.handle_logging("Another log entry for data validation.")

    logger1 = AppLogger("data_ingestion")
    print(logger1.logs_phase_path)
    logger1.handle_logging("This is a test log message for data ingestion.",level= logging.ERROR)

    # Append more log messages to the same log file
    logger1.handle_logging("Another log entry for data ingestion.")
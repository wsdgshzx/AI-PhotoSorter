import logging
import os
import datetime
from enum import Enum

class ConsoleColor(Enum):
    RED = 31
    GREEN = 32
    YELLOW = 33
    BLUE = 34
    MAGENTA = 35
    CYAN = 36

# Create a custom Logger
class CustomLogger:
    loggers = {}

    def __new__(cls, name, *args, **kwargs):
        # Use singleton pattern
        if name not in cls.loggers:
            cls.loggers[name] = super(CustomLogger, cls).__new__(cls)
        return cls.loggers[name]

    def __init__(self, name, queue = None):
        if not hasattr(self, 'file_logger'):
            self.file_logger = logging.getLogger(name)
            self.file_logger.setLevel(logging.DEBUG)

            # Configure file handler
            log_dir = os.path.join(os.getcwd(), 'log')
            os.makedirs(log_dir, exist_ok=True)

            project_name = os.path.basename(os.path.dirname(os.getcwd()))
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_file_path = os.path.join(log_dir, f"{project_name}_{current_time}.log")
            fh = logging.FileHandler(log_file_path, encoding='utf-8')
            fh.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(file_formatter)
            self.file_logger.addHandler(fh)
            self.file_logger.propagate = False

            self.queue = queue
            if self.queue:
                print(f"queue: {self.queue}")
                

    def _log(self, level, msg, color=None):
        # Use print for console output and apply color codes
        if color:
            color_code = color.value
            print(f"\033[{color_code}m{msg}\033[0m")
        else:            
            print(msg)

        if self.queue:
            self.queue.put(msg)

        # File log keeps the original message
        self.file_logger.log(level, msg)

    def debug(self, msg, color=None):
        self._log(logging.DEBUG, msg, color)

    def info(self, msg, color=None):
        self._log(logging.INFO, msg, color)

    def warning(self, msg, color=None):
        self._log(logging.WARNING, msg, color)

    def error(self, msg, color=None):
        self._log(logging.ERROR, msg, color)

    def critical(self, msg, color=None):
        self._log(logging.CRITICAL, msg, color)

    @staticmethod
    def print(msg, color=None):
        if color:
            color_code = color.value
            print(f"\033[{color_code}m{msg}\033[0m")
        else:
            print(msg)

if __name__ == '__main__':
# Example usage
    project_name = "my_project"
    log_file_name = "my_log.log"
    log_dir = "path/to/log/directory"
    log_file_path = os.path.join(log_dir, log_file_name)

    custom_logger = CustomLogger(project_name, log_file_path)
    custom_logger.info("This is an info message", color=ConsoleColor.RED)  # Use colors with descriptive names
    custom_logger.info("Another info message")  # Default color
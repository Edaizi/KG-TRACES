import os
import time
import logging
from logging.handlers import RotatingFileHandler
import sys
import atexit 

try:
    import colorama
    from colorama import Fore, Style
    colorama.init()
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False

class ColoredFormatter(logging.Formatter):

    COLOR_MAP = {
        logging.DEBUG: Fore.CYAN if COLORAMA_AVAILABLE else '',
        logging.INFO: Fore.GREEN if COLORAMA_AVAILABLE else '',
        logging.WARNING: Fore.YELLOW if COLORAMA_AVAILABLE else '',
        logging.ERROR: Fore.RED if COLORAMA_AVAILABLE else '',
        logging.CRITICAL: Fore.MAGENTA if COLORAMA_AVAILABLE else '',
    }

    RESET = Style.RESET_ALL if COLORAMA_AVAILABLE else ''

    def format(self, record):
        color = self.COLOR_MAP.get(record.levelno, '')
        message = super().format(record)
        if color:
            message = f"{color}{message}{self.RESET}"
        return message

def setup_logger(logger_name, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = True

    if not logging.getLogger().hasHandlers():
        main_script = sys.argv[0]
        if not main_script:
            main_script_dir = os.getcwd()
            main_script_name = 'unknown'
        else:
            main_script_path = os.path.abspath(main_script)
            main_script_dir = os.path.dirname(main_script_path)
            main_script_name = os.path.splitext(os.path.basename(main_script_path))[0]

        log_dir = os.path.join('logs', main_script_name)
        os.makedirs(log_dir, exist_ok=True)


        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        timestamp_log_file = os.path.join(log_dir, f'{timestamp}.log')

        latest_log_file = os.path.join(log_dir, 'latest.log')


        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        colored_formatter = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


        timestamp_handler = RotatingFileHandler(
            timestamp_log_file,
            maxBytes=1000*1024*1024,  
            backupCount=5,
            encoding='utf-8'
        )
        timestamp_handler.setFormatter(formatter)
        timestamp_handler.setLevel(level)
        logging.getLogger().addHandler(timestamp_handler)


        latest_handler = logging.FileHandler(
            latest_log_file,
            mode='w',               
            encoding='utf-8'
        )
        latest_handler.setFormatter(formatter)
        latest_handler.setLevel(level)
        logging.getLogger().addHandler(latest_handler)


        console_handler = logging.StreamHandler()
        console_handler.setFormatter(colored_formatter)
        console_handler.setLevel(level)
        logging.getLogger().addHandler(console_handler)


        def append_log_paths():
            log_paths = f"\nSave log file to:\n{timestamp_log_file}\n{latest_log_file}\n"
            try:
     
                with open(timestamp_log_file, 'a', encoding='utf-8') as f:
                    f.write(log_paths)
           
                with open(latest_log_file, 'a', encoding='utf-8') as f:
                    f.write(log_paths)
              
                print(log_paths)
            except Exception as e:
               
                print(f"Failed to append log paths: {e}")


        atexit.register(append_log_paths)

    return logger



def test_setup_setup_logger():
    logger = setup_logger('tool_test')
    logger.info('This is a log info')


if __name__ == "__main__":
    test_setup_setup_logger()

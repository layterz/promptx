import json
import logging
import textwrap
import colorama

colorama.just_fix_windows_console()


FORMAT_INSTRUCTIONS = logging.DEBUG + 1
CONTEXT = logging.DEBUG + 2
EXAMPLES = logging.DEBUG + 3
METRICS = logging.DEBUG + 4
HISTORY = logging.DEBUG + 5
INSTRUCTIONS = logging.INFO + 1
INPUT = logging.INFO + 2
OUTPUT = logging.INFO + 3

logging.addLevelName(FORMAT_INSTRUCTIONS, "FORMAT INSTRUCTIONS")
logging.addLevelName(CONTEXT, "CONTEXT")
logging.addLevelName(EXAMPLES, "EXAMPLES")
logging.addLevelName(INSTRUCTIONS, "INSTRUCTIONS")
logging.addLevelName(INPUT, "INPUT")
logging.addLevelName(OUTPUT, "OUTPUT")
logging.addLevelName(METRICS, "METRICS")
logging.addLevelName(HISTORY, "HISTORY")

class NotebookFormatter(logging.Formatter):

    colors = {
        FORMAT_INSTRUCTIONS: colorama.Fore.CYAN,
        CONTEXT: colorama.Fore.BLACK + colorama.Back.YELLOW,
        EXAMPLES: colorama.Fore.MAGENTA,
        METRICS: colorama.Fore.WHITE + colorama.Style.DIM,
        HISTORY: colorama.Back.WHITE + colorama.Fore.BLACK,
        INSTRUCTIONS: colorama.Fore.YELLOW,
        INPUT: colorama.Fore.BLUE,
        OUTPUT: colorama.Fore.GREEN,
        logging.ERROR: colorama.Fore.RED,
        logging.INFO: colorama.Fore.WHITE,
        logging.DEBUG: colorama.Style.DIM,
    }

    def __init__(self, width=80, **kwargs):
        super().__init__(**kwargs)
        self.width = width

    def format(self, record) -> str:
        color = self.colors.get(record.levelno, None)
        msg = f'{color}{record.getMessage()}{colorama.Style.RESET_ALL}'
        return '\n'.join([
            textwrap.fill(line.strip(), width=self.width).ljust(self.width, ' ')
            for line in msg.split('\n')
        ])


class JSONLogFormatter(logging.Formatter):
    def format(self, record):
        msg = super().format(record)
        log_entry = {
            "timestamp": record.created,
            "level": record.levelname,
            "message": msg,
            "filename": record.filename,
            "lineno": record.lineno,
            "funcName": record.funcName,
        }
        return json.dumps(log_entry)
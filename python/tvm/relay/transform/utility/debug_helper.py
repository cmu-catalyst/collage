import sys
import datetime
import sys
import os
import logging


# With more argumentsc
def printe(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, level):
       self.logger = logger
       self.level = level
       self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())
        # sys.stdout.flush()
        # sys.stderr.flush()

    def flush(self):
        pass

# Add new logging level
MYINFO = 35
logging.addLevelName(MYINFO, 'MYINFO')
def myinfo(self, message, *args, **kws):
    self.log(MYINFO, message, *args, **kws)
logging.Logger.myinfo = myinfo

def setup_logging(task_name, net_name, hw_name, logging_level=logging.WARNING):
    date_now = datetime.datetime.now()
    this_code_path = os.path.dirname(os.path.abspath(__file__))
    date_now = date_now.strftime("%m-%d-%H:%M")
    file_path = f"{this_code_path}/../logs/exp_logs/{task_name}_{hw_name}_{net_name}_{date_now}"

    logging.basicConfig(filename=file_path, level=logging_level,
                        format='%(asctime)s:[%(levelname)s] %(message)s')

    log = logging.getLogger('logger')


    sys.stdout = StreamToLogger(log, MYINFO)
    sys.stderr = StreamToLogger(log, logging.ERROR)

# def printe(msg):
#     print(msg, file=sys.stderr)

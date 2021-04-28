import logging, sys

class Logger():

    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
    
    
    def log(self, msg):
        self.logger.info(msg)
    
    def log_error(self, msg):
        self.logger.warn(msg)
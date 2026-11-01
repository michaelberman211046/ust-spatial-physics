import sys

class Logger(object):
    def __init__(self, filename, append=False):
        self.terminal = sys.stdout
        mode = "a" if append else "w"
        self.log = open(filename, mode)
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # Required by file-like stream interfaces.
        pass











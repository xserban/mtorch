from pathlib import Path
from py_elasticinfra.elk.elastic import Indexer
from py_elasticinfra.runner import Runner
from py_elasticinfra.utils.parse_config import ConfigParser


class InfraLogger:
    def __init__(self, config, logger):
        print("[INFO] \t Initializing Infrastructure Logger ...")
        self.config = ConfigParser(config)
        try:
            self.es = Indexer(config, logger)
            self.es.connect()
            self.es.create_index()
        except Exception as excpt:
            raise ("[ERROR] Could not Config "
                   "Infrastructure "
                   "Logger: {}".format(excpt))
        if self.es:
            self.runner = Runner(self.config, self.es)

    def start(self):
        if self.runner:
            print("[INFO] \t Starting "
                  "Infrastructure Logging.")
            self.runner.run_background()

    def stop(self):
        if self.runner:
            print("[INFO] \t Stopping "
                  "Infrastructure Logging.")
            self.runner.stop_background()

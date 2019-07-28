"""This class is a wrapper around sacred experiment, containing
an instance of the experiment and other details.
It is a Singleton so we can import it from different files and
get the same instance.
"""
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred import SETTINGS
from utils.singleton import Singleton

ex = Experiment()


class Sacred(metaclass=Singleton):
    def __init__(self, config):
        super(Sacred, self).__init__()
        print('[INFO] \t Configuring Sacred Logger')
        global ex
        ex = Experiment(config['name'])

        self.sacred_config = config['trainer']['sacred_logs']
        self._configure_observer()
        self._set_settings()
        self.add_config(config)

    def _configure_observer(self):
        ex.observers.append(MongoObserver.create(
            url=self.sacred_config['mongo_url'],
            db_name=self.sacred_config['db_name']
        ))

    def _set_settings(self):
        SETTINGS['CAPTURE_MODE'] = self.sacred_config['capture_mode']

    def add_config(self, config):
        global ex
        ex.add_config(config.json_config)

    @ex.main
    def run(self, main, config):
        main(config)

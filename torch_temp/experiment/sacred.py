import os

from torch_temp.utils import Singleton
from sacred.observers import MongoObserver
from sacred import SETTINGS


class Sacred(metaclass=Singleton):
    def __init__(self, experiment, config, auto_config=False, parent_folder=None):
        """Initializes Sacred Singleton used to configure all s
        Sacred related settings
        :param experiment: sacared Experiment object
        :param config: config dic
        :param auto_config: if true, all settings from Sacred are
            configured on init
        """
        super().__init__()
        self.ex = experiment
        self.config = config

        if auto_config is True:
            self.add_config(config)
            self.add_mongo_observer(config)
            self.add_settings(config['settings'])

    def add_config(self, config):
        print('[INFO] \t Setting Sacred Config File.')
        self.ex.add_config(config)

    def add_mongo_observer(self, config):
        print('[INFO] \t Configuring Sacred MongoDB Observer.')
        self.ex.observers.append(MongoObserver.create(
            url=config["mongo_url"],
            db_name=config["db_name"]))

    def add_settings(self, settings):
        print('[INFO] Added all Sacred Settings.')
        for key, value in settings.items():
            if isinstance(value, str):
                SETTINGS[key] = value
            else:
                pass
                # TODO: configure iterative parsing of sacred config

    def add_all_files(self, parent_folder):
        if self.config['save_files']:
            print('[INFO] \t Indexing all Source Files in Sacred MongoDB.')
            file_set = set()
            for dir_, _, files in os.walk(parent_folder):
                for file_name in files:
                    if file_name.endswith((".py")):
                        rel_dir = os.path.relpath(dir_, parent_folder)
                        rel_file = os.path.join(rel_dir, file_name)
                        file_set.add(rel_file)

            for file_name in file_set:
                print(parent_folder + file_name)
                try:
                    # self.ex.add_source_file(parent_folder + file_name)
                    self.ex.add_resource(parent_folder + file_name)
                except Exception as e:
                    print(e)
                    raise('File not found')

import os

from pathlib import Path
from sacred import SETTINGS
from sacred.observers import MongoObserver
from core.utils import read_json


class Sacred():
    def __init__(self, experiment, config, auto_config=False):
        """Initializes Sacred Experiments
        Sacred related settings
        :param experiment: sacared Experiment object
        :param config: config dic
        :param auto_config: if true, all settings from Sacred are
            configured on init
        """
        super().__init__()
        self.ex = experiment
        self.ex.add_config(config)
        self.config = config["logging"]["sacred_logs"]

        if auto_config is True:
            self.add_mongo_observer(self.config)
            self.add_settings(self.config["settings"])

    def add_mongo_observer(self, config=None):
        print("[INFO][LOGS] \t Configuring Sacred MongoDB Observer.")
        if config is None:
            config = self.config
        self.ex.observers.append(MongoObserver.create(
            url=config["mongo_url"],
            db_name=config["db_name"]))

    def add_settings(self,
                     settings="core/logger/sacred_logger_config.json"):
        print("[INFO][LOGS] \t Configuring Sacred Settings.")
        if isinstance(settings, str):
            log_config = Path(settings)
            if log_config.is_file():
                config = read_json(log_config)
                config = config["settings"]
            else:
                raise ValueError("Incorrect settings path")
        else:
            config = settings
        for key, value in config.items():
            if isinstance(value, str):
                SETTINGS[key] = value
            else:
                pass
                # TODO: configure iterative parsing of sacred config

    def add_all_files(self, parent_folder):
        print("[INFO][LOGS] \t Indexing Extra Source Files in Sacred MongoDB.")
        file_set = set()
        for dir_, _, files in os.walk(parent_folder):
            for file_name in files:
                if file_name.endswith((".py")):
                    rel_dir = os.path.relpath(dir_, parent_folder)
                    rel_file = os.path.join(rel_dir, file_name)
                    file_set.add(rel_file)

        for file_name in file_set:
            try:
                self.ex.add_resource(parent_folder + file_name)
            except Exception as exception:
                raise exception

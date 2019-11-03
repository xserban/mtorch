"""
  This module runs several experiments based on a folder with config files.
  It prioritizes experiments which use all gpus, and afterwards runs
  experiments on custom GPUs.
  Currently the order is sequential: first the experiments on multiple gpus
  run and after the parallel experiments on custom gpus. In the future it
  would be nice to interleave experiments.

  This module is still experimental, it was only tested on 2 gpus,
  so use it with care :-)
"""
import os
import torch

from core.utils.util import read_json, remove_from_dic
from core.experiment.runner import Runner
from threading import Thread


class MRunners:
    def __init__(self, configs):
        self.m_gpus, self.cgpus = self.sort_conf(configs)
        self.run_all()

    def sort_conf(self, configs):
        multiple_gpus = [x for x in configs if x.config["host"]
                         ["gpu_settings"]["multiple_gpus"]["do"] is True]

        custom_gpus = [x for x in configs if x.config["host"]
                       ["gpu_settings"]["custom_gpu"]["do"] is True]

        # make sure there are no experiments in both groups
        # and if they are run them only on multiple gpus
        intersect = [x for x in multiple_gpus if x in custom_gpus]
        if len(intersect) > 0:
            custom_gpus = remove_from_dic(intersect, custom_gpus)
        return multiple_gpus, custom_gpus

    def run_all(self):
        print("[INFO] \t Running all experiments on multiple GPUs")
        self.run_multiple_gpus()
        print("[INFO] \t Running all experiments on custom GPUs")
        self.run_custom_gpus()

    def run_multiple_gpus(self):
        for cnf in self.m_gpus:
            run = Runner(cnf)
            run.run_experiment()

    def run_custom_gpus(self):
        groups = self.group_runs()
        sgroups = sorted(groups, key=lambda g: g["len_ids"], reverse=True)
        # pair tasks on disjoing gpus
        tasks = []
        for task in sgroups:
            t, _ = self.find_pairs(task, sgroups)
            tasks.append(t)

        for task in tasks:
            threads = []
            for c in task["configs"]:
                r = Runner(c)
                t = Thread(target=r.run_experiment)
                threads.append(t)
            # Start all threads
            for t in threads:
                t.start()
            # Wait for all of them to finish
            # TODO: change this so when a thread
            # stops a new task is performed
            # with the current implementation, a task finishes
            # only when all config files are ran. So one experiment
            # may "wait" for the others and not use resources wisely
            for t in threads:
                t.join()

    def find_pairs(self, task, sgroups):
        max_gpu = torch.cuda.device_count()
        options = [x for x in sgroups if (x["len_ids"]+task["len_ids"])
                   <= max_gpu and x["str_ids"] != task["str_ids"]]
        # filter out common gpus
        options = [o for o in options if not any(
            item in o["ids"] for item in task["ids"])]
        # sort by the max number of gpu occupancy
        options = sorted(
            options, key=lambda o: (o["len_ids"]+task["len_ids"]))

        if len(options) > 0:
            # merge tasks
            task["configs"] += options[0]["configs"]
            task["str_ids"] += " " + options[0]["str_ids"]
            # task["ids"] += options[0]["ids"]
            task["len_ids"] += options[0]["len_ids"]
            sgroups.pop(sgroups.index(options[0]))
            # search for another concurrent task
            if ((task["len_ids"]) < max_gpu):
                task, sgroups = self.find_pairs(task, sgroups)

        return task, sgroups

    def group_runs(self):
        grps = []
        for cnf in self.cgpus:
            ids = cnf["host"]["gpu_settings"]["custom_gpu"]["ids"]
        # ex = [x for x in grps if x['str_ids'] == str(ids)]

        # if len(ex) > 0:
        # ex[0]["configs"].append(cnf)
        # else:
            grps.append({
                "str_ids": str(ids),
                "ids": ids,
                "len_ids": len(ids),
                "configs": [cnf]
            })
        return grps

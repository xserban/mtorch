from sacred import Experiment
from sacred import SETTINGS

# Currently the discover sources flag must be set here.
# Please see the issue on github:
# https://github.com/IDSIA/sacred/issues/546
SETTINGS["DISCOVER_SOURCES"] = "dir"
ex = Experiment("")


@ex.main
def run_exp():
    # the main_ method is set in a runner
    ex.main_()

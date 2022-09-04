import os

from michal.michal_utils import is_running_on_polyaxon


DIRPATH_ROOT = "/cluster-polyaxon/users/molesz/" if is_running_on_polyaxon() else ""
DIRPATH_CHECKPOINTS = os.path.join(DIRPATH_ROOT, "checkpoints")

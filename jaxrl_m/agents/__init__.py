from .continuous.bc import BCAgent
from .continuous.gc_bc import GCBCAgent
from .continuous.lc_bc import LCBCAgent
from .continuous.glc_bc import GLCBCAgent
from .continuous.gc_iql import GCIQLAgent
from .continuous.iql import IQLAgent
from .continuous.gc_ddpm_bc import GCDDPMBCAgent
from .continuous.stable_contrastive_rl import StableContrastiveRLAgent

agents = {
    "gc_bc": GCBCAgent,
    "lc_bc": LCBCAgent,
    "glc_bc": GLCBCAgent,
    "gc_iql": GCIQLAgent,
    "gc_ddpm_bc": GCDDPMBCAgent,
    "bc": BCAgent,
    "iql": IQLAgent,
    "stable_contrastive_rl": StableContrastiveRLAgent,
}

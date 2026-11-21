"""A-role observation contract repair, independent of frozen dynamic chain.

The upstream RecordsAgent can compare two *currently legal* source packages,
but the original A observation omitted the public future privacy-channel
setup fee and public task cost. That information affects which package is
long-horizon optimal and is already visible to the later Coordinator.
A cannot learn a correctly conditional future-cost policy if its observation
is missing a legitimate task constraint.

This changes only the ACTOR observation at upstream A. It never supplies
hidden receiver truth, optimal action, future Q, privacy verdict, or graph
closure; the signed source tools and native privacy/database state transitions
remain entirely untouched.
"""
from __future__ import annotations
from copy import deepcopy
from .upstream_downstream_chain_pilot import ChainPilotEnv

class ChainPilotEnvPublicContract(ChainPilotEnv):
    def observation(self,state):
        obs=super().observation(state)
        if state.phase in ("A_READ","A_SEND") and not state.native.done:
            contract=state.native.business.get("privacy_resource_contract")
            if not contract:
                raise ValueError("Downstream public privacy resource contract unavailable")
            obs["privacy_resource_contract"]=deepcopy(contract)
            # The actor already sees the future required business checks and
            # source package per-option processing costs; its missing budget
            # / preflight setup fee is now transparently available too.
            assert set(("private_channel_setup_steps","skip_setup_steps",
                        "task_budget","step_cost"))<=set(contract)
        return obs

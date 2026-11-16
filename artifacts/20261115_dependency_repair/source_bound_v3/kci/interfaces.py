"""Environment adapters for external agent frameworks and CI datasets."""
from __future__ import annotations
from typing import Protocol
from .contracts import State,Observation,Action,StepAudit

class CIEnvironment(Protocol):
    # Identify both implementation and fixed semantic configuration; change this
    # version when norms, transitions, observations, or execution costs change.
    version: str
    def observation(self,state:State,actor:str)->Observation:
        """Permission-filtered observation; all private oracle fields excluded."""
        ...
    def transition(self,state:State,action:Action)->tuple[State,StepAudit]:
        """Execute one action in a sandbox and audit this *delivered* transition.

        The state returned must be a new object; violations are recorded, not erased.
        A failed delivery MUST NOT change recipient knowledge.
        """
        ...

class SemanticVerifier(Protocol):
    def verify(self,message_body:str,claimed_fact_ids:tuple[str,...])->StepAudit:
        """Return VALID/UNKNOWN with verified semantic evidence.

        Model claims are untrusted. UNKNOWN must not be rewarded as safe.
        """
        ...

class TaskOutcomeVerifier(Protocol):
    def task_success(self,state:State)->bool|None:
        """None means UNKNOWN; a language model's self-report is not a task oracle."""
        ...

class ExternalPolicy(Protocol):
    version:str
    def choose(self,observation:Observation,seed:int)->Action:
        """Use frozen model weights/version and only authorized observation."""
        ...

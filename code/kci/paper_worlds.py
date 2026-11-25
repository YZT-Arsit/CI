"""Replayable V8 TRAIN environments for the ACL draft. Raw sources stay local."""
from copy import copy
from dataclasses import asdict
from fractions import Fraction
from functools import lru_cache
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys

from . import v4_phase2_workflows as phase
from .diverse_ci_branch_env import FINAL, SLOT_LABELS


def canon(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def key(x):
    return sha256(canon(x).encode()).hexdigest()


def filehash(path):
    with Path(path).open("rb") as f:
        return sha256(f.read()).hexdigest()


def lines(path):
    with path.open(encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]


class PhaseWorld:
    def __init__(self, spec, authorized=False, reorder=False, swap=False, variant="risky"):
        self.env = phase.Phase2Env(spec, authorized=authorized, reorder=reorder, package_swap=swap, variant=variant)
        self.root = spec.id
        self.id = key([asdict(spec), authorized, reorder, swap, variant])
        self.norm = "phase2_scoped_protected_association"
        self.initial = self.env.initial
        self.reference_version = "visible_phase_rules_v1"

    def observe(self, s):
        return self.env.observation(s)

    def reference(self, ob):
        actions = ob["legal_actions"]
        if ob["stage"] == "preflight":
            return "OPEN_PRIVATE_CHANNEL"
        if ob["stage"] == "release":
            rules = ob["public_inference_rules"]
            facts = {SLOT_LABELS.get(e["fact"], e["fact"]) for e in ob["delivered_message_log"]
                     if e["recipient"] == ob["recipient"] and e["subject"] == ob["current_case_subject"]}
            facts.add(ob["signed_source_tool_record"]["slot"])
            while True:
                updated = facts | {r["entails"] for r in rules if set(r["requires"]) <= facts}
                if updated == facts:
                    break
                facts = updated
            if "protected association" in facts and "No authorization exception" in ob["privacy_policy"]:
                return next(a for a in ("VERIFY_PRIVATE_TARGET", "SEND_RAW_BROKER", "ABORT") if a in actions)
            return "SEND_RAW_TARGET"
        return actions[0]

    def branches(self, s, a):
        n, report = self.env.transition(s, a)
        events = n.ledger[len(s.ledger):]
        direct = any(e["fact"] == FINAL and not self.env._norm_authorized(e["recipient"]) for e in events)
        delta = report.get("knowledge_delta", {})
        inferred = any(FINAL in facts and not self.env._norm_authorized(r) for r, facts in delta.items())
        if report["verdict"] in ("VALID", "VIOLATION") and (direct or inferred) != (report["verdict"] == "VIOLATION"):
            raise ValueError("Delivery evidence and native CI verdict disagree")
        return [(Fraction(1), n, dict(verdict=report["verdict"], direct=direct, inferred=inferred,
                                      delta=delta, events=events, reason=report.get("reason")))]

    @staticmethod
    def terminal(s):
        return s.done, s.success, s.steps, s.aborted

    def omitted(self, s):
        return bool(set(self.env.spec.required) - set(s.verified))

    def downstream(self, ob):
        return ob["stage"] not in ("A_READ", "A_SEND")


class RelWorld:
    def __init__(self, mod, spec, source, authorized=False):
        self.env = mod.RelationalInferenceEnv(spec, source, seed=0, preauthorized=authorized)
        self.mod = mod
        self.root = self.env.root
        self.id = self.env.env_id
        self.norm = "v8_stipulated_source_join_consent"
        policy = mod.ReferencePolicy()
        self.reference_version = policy.version
        self.reference = policy.choose
        self.initial = self.env.initial_state

    def observe(self, s):
        ob = self.env.observation(s)
        ob.pop("oracle_aggregate_result_NOT_given", None)
        return ob

    def branches(self, s, a):
        env, mod = self.env, self.mod
        kind = a["kind"]
        p = None
        if kind == mod.CONSENT:
            p = env.policy["consent_grant_probability"]
        elif kind == mod.PRIVATE and env._stage(s) == "protected":
            p = env.policy["private_protected_channel_unavailable_probability"]
        # Enumerate the public chance law at the transition where it occurs.
        # No future draw/seed is included in the actor's observation.
        choices = [(Fraction(1), None)] if p is None else [
            (Fraction(str(p)), 0.0), (1-Fraction(str(p)), 1.0)]
        out = []
        for mass, draw in choices:
            if not mass:
                continue
            branch = copy(env)
            if draw is not None:
                branch._random_uniform = lambda *args: draw
            n, rep = branch.transition(s, a)
            if rep["verdict"] not in (mod.VALID, mod.VIOLATION):
                out.append((mass, n, dict(verdict=rep["verdict"], direct=False, inferred=False,
                                          delta={}, events=[], reason=rep.get("reason"))))
                continue
            before = {canon(r) for r in rep["actual_recipient_K_before"]["inferred_protected_associations"]}
            after = {canon(r) for r in rep["actual_recipient_K_after"]["inferred_protected_associations"]}
            added = [json.loads(r) for r in sorted(after-before)]
            events = n.event_log[len(s.event_log):]
            inferred = bool(added and not n.authorized_relation)
            if inferred != (rep["verdict"] == mod.VIOLATION):
                raise ValueError("V8 join proof and native verdict disagree")
            # Both projection routes are expressly permitted as individual flows in this source.
            def received(ledger):
                # A copied or re-sent proposition yields only one knowledge fact.
                return {canon([ev["source_stage"], ev["content"]]) for ev in ledger
                        if ev["status"] == "DELIVERED"}
            delta = {}
            for who, old, new in ((mod.RECIPIENT, s.recipient_receipts, n.recipient_receipts),
                                  (mod.BROKER, s.broker_receipts, n.broker_receipts)):
                delta[who] = {"received": [json.loads(v) for v in sorted(received(new) - received(old))]}
            delta[mod.RECIPIENT]["inferred_protected_associations"] = added
            out.append((mass, n, dict(verdict=rep["verdict"], direct=False, inferred=inferred,
                                      delta=delta, events=events)))
        return out

    @staticmethod
    def terminal(s):
        # Native 'success' folds privacy in. Utility separately measures the actual submission.
        correct = s.business_result is not None and s.submitted_answer is s.business_result
        return s.done, correct, s.cost, s.aborted

    def omitted(self, s):
        # The requested output is the threshold answer. A wrong answer was submitted.
        return s.submitted_answer is None

    def downstream(self, ob):
        return ob["role_stage"] != "public"


@lru_cache(maxsize=1)
def relational_module(path):
    spec = spec_from_file_location("kci_paper_v8_relational", path)
    mod = module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def worlds(research, families=("phase2","relational")):
    v8 = research/"kci_supervision_alignment_20261123/deliverables/KCI_DIVERSE_V8_INTERNAL"
    hashes = json.loads((v8/"FILE_SHA256.json").read_text(encoding="utf8"))
    def source_rows(stage):
        name = "train/"+stage+".jsonl"
        if filehash(v8/name) != hashes[name]:
            raise ValueError("V8 source file changed: "+name)
        rows = lines(v8/name)
        if any(r["split"].lower() != "train" for r in rows):
            raise ValueError("Unexpected non-TRAIN source")
        return rows
    if "phase2" in families:
        roots = {r["original_independent_root"] for r in source_rows("PHASE2_REAL4B_DPR")}
        specs = {s.id:s for s in phase.make_workflows() if s.split.lower()=="train"}
        for root in sorted(roots):
            for authorized, reorder, swap, variant in [(False,False,False,"risky"),
                    (False,False,True,"risky"), (True,False,False,"risky"),
                    (False,True,False,"risky"), (False,False,False,"hold")]:
                yield PhaseWorld(specs[root], authorized, reorder, swap, variant)
    if "relational" in families:
        roots = {r["case_root_id"] for r in source_rows("REL_SQL_DYNAMIC_CI_STRICT")}
        d = research/"kci_diverse_relational_20261124"
        src = research/"kci_public_refs_20261119/MultiAgentPrivacy/data"
        specs = d/"data/PREREG_RELATIONAL_TASKS_V2/PREREG_TRAIN_MECHANISMS_WITH_QUERIES.jsonl"
        envfile = d/"code/relational_env.py"
        pin = json.loads((d/"data/PREREG_TRAIN_REFERENCE_EVAL_V1/PROTOCOL.json").read_text(encoding="utf8"))
        if filehash(specs)!=pin["source_train_mechanism_sha256"] or filehash(envfile)!=pin["environment_code_sha256"]:
            raise ValueError("Native relational source or verifier changed")
        for spec in lines(specs):
            if spec["case_root_id"] not in roots:
                continue
            p = src/spec["source_file"]
            if spec["split"] != "train" or filehash(p)!=spec["source_sha256"]:
                raise ValueError("Relational world provenance mismatch")
            source = json.loads(p.read_text(encoding="utf8"))
            for auth in (False,True):
                yield RelWorld(relational_module(envfile),spec,source,auth)


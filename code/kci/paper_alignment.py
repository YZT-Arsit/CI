"""Algorithm 1 on bounded, replayable worlds. The actor receives only observe(s)."""
from collections import defaultdict
from dataclasses import asdict
from fractions import Fraction
from itertools import combinations
import random

from .paper_worlds import canon, key

FIELDS = ("direct", "inferred", "any_ci", "unknown", "success", "stsr", "safe_success_cost_mass",
          "aborted", "omitted_at_terminal", "direct_and_inferred")


def actor(ob):
    return ob.get("acting_agent", ob.get("actor"))


def prompt(ob):
    # Source split and identifiers are evaluator bookkeeping. Retain role-visible receipts.
    public = {k:v for k,v in ob.items() if k not in ("split","case_root_id","mechanism_id","source_attribution")}
    return canon(public)


def metrics(values):
    out = {k:float(v) for k,v in zip(FIELDS, values)}
    out["cost_given_safe_success"] = float(values[6]/values[5]) if values[5] else None
    return out


class Alignment:
    def __init__(self, world, policy, version, horizon=24):
        self.world, self.policy, self.version = world, policy, version
        self.horizon = horizon
        self.cache = {}
        self.states = {}
        self.snapshots = {}
        self.pairs = {}
        self.failures = []

    def sid(self, s):
        return key([self.world.id, asdict(s)])

    def outcome(self, s, left, name, direct=False, inferred=False):
        """Exact expectation under one frozen policy and the native event law."""
        tag = self.sid(s), left, name, direct, inferred
        if tag in self.cache:
            return self.cache[tag]
        done, utility, cost, aborted = self.world.terminal(s)
        if done or left == 0:
            bad = direct or inferred
            success = done and utility
            safe = success and not bad
            result = (tuple(map(Fraction,(direct,inferred,bad,not done and not bad,success,safe,
                                          str(cost) if safe else 0,aborted,
                                          done and self.world.omitted(s),direct and inferred))), [] if safe else None)
        else:
            ob = self.world.observe(s)
            a = (self.world.reference if name=="reference" else self.policy)(ob)
            result = self.follow(s, a, left, name, direct, inferred)
        self.cache[tag] = result
        return result

    def follow(self, s, a, left, name, direct=False, inferred=False):
        totals = [Fraction(0)]*len(FIELDS)
        witness = None
        branches = self.world.branches(s,a)
        if sum(m for m,_,_ in branches)!=1:
            raise ValueError("Transition mass is not one")
        for mass,n,audit in branches:
            if audit["verdict"] not in ("VALID","VIOLATION"):
                v = (direct,inferred,direct or inferred,not (direct or inferred),0,0,0,0,0,direct and inferred)
                suffix = None
            else:
                v,suffix = self.outcome(n,left-1,name,direct or audit["direct"],inferred or audit["inferred"])
            for i,x in enumerate(v):
                totals[i] += mass*x
            if suffix is not None and witness is None:
                witness = [dict(before=self.sid(s),action=a,after=self.sid(n),mass=str(mass))]+suffix
        return tuple(totals),witness

    def evaluate(self,s,left):
        sid = self.sid(s)
        if (sid,left) in self.states:
            return self.states[sid,left]
        ob = self.world.observe(s)
        reports = []
        for a in ob["legal_actions"]:
            branches = self.world.branches(s,a)
            immediate = {r["verdict"] for _,_,r in branches}
            status = next(iter(immediate)) if len(immediate)==1 else "UNKNOWN"
            row = dict(action=a, immediate=status, transitions=[
                dict(mass=str(m),after=self.sid(n),audit=r) for m,n,r in branches])
            if status=="VALID":
                for name in ("current","reference"):
                    v,wit = self.follow(s,a,left,name)
                    row[name] = metrics(v)
                    if name=="reference":
                        row["witness"] = wit
                        row["vector"] = v
                row["evidence_class"] = ("exact_within_horizon" if not row["vector"][3] else
                                         "verified_witness_only" if row["witness"] is not None else "unknown")
            else:
                row["evidence_class"] = "verified_violation" if status=="VIOLATION" else "unknown"
            reports.append(row)
        self.states[sid,left] = reports
        self.snapshots[sid] = dict(snapshot=asdict(s),actor=actor(ob),observation_id=key(prompt(ob)))
        return reports

    @staticmethod
    def preference(a,b):
        """A successful witness proves sufficiency. Ranking uses exact fixed-policy values."""
        for good,bad in ((a,b),(b,a)):
            if good["immediate"]=="VALID" and good.get("witness") is not None and bad["immediate"]=="VIOLATION":
                return good,bad,"ci"
        if a["immediate"]!="VALID" or b["immediate"]!="VALID":
            return None
        av,bv = a["vector"],b["vector"]
        if av[3] or bv[3]:
            return None
        def rank(v):
            return (-v[2],v[5],-v[6]/v[5] if v[5] else 0)
        if rank(av)==rank(bv):
            return None
        good,bad = (a,b) if rank(av)>rank(bv) else (b,a)
        if good["witness"] is None:
            return None
        return good,bad,"future"

    def pair(self,s,left,good,bad,source,trace_id=None):
        ob = self.world.observe(s)
        row = dict(source=source,root=self.world.root,world=self.world.id,state=self.sid(s),remaining=left,
                   actor=actor(ob),observation_id=key(prompt(ob)),prompt=prompt(ob),norm=self.world.norm,
                   chosen=good["action"],rejected=bad["action"],weight=1.0,
                   evidence="verified_violation_and_witness" if bad["immediate"]=="VIOLATION" else "exact_fixed_reference",
                   current_policy=self.version,reference_policy=self.world.reference_version,
                   trace_id=trace_id)
        pair_id=key([source,row["state"],left,row["chosen"],row["rejected"]])
        self.pairs.setdefault(pair_id, row)

    def visit(self,seed,which="current",first=None,mine=True):
        s = self.world.initial()
        path, visited = [],[]
        direct = inferred = False
        rng = random.Random(seed)
        for left in range(self.horizon,0,-1):
            if self.world.terminal(s)[0]:
                break
            ob=self.world.observe(s)
            if first is not None and first[0](ob):
                a=first[1](ob)
                first=None
            else:
                a=(self.policy if which=="current" else self.world.reference)(ob)
            if a not in ob["legal_actions"]:
                raise ValueError("Actor returned an unavailable action")
            if mine and not (direct or inferred):
                rs=self.evaluate(s,left)
                for x,y in combinations(rs,2):
                    pref=self.preference(x,y)
                    if pref:self.pair(s,left,*pref)
                visited.append((s,left,rs,a))
            branches=self.world.branches(s,a)
            u=rng.random()
            for mass,n,audit in branches:
                u-=float(mass)
                if u<0:break
            path.append(dict(state=self.sid(s),observation_id=key(prompt(ob)),actor=actor(ob),action=a,
                             after=self.sid(n),audit=audit))
            s=n
            if audit["verdict"] not in ("VALID","VIOLATION"):break
            direct |= audit["direct"]
            inferred |= audit["inferred"]
        done,utility,cost,aborted=self.world.terminal(s)
        bad=direct or inferred
        unresolved=not done or any(x["audit"]["verdict"] not in ("VALID","VIOLATION") for x in path)
        trace_id=key([self.world.id,self.version,which,seed,path])
        if mine and which=="current" and (bad or (done and not utility and not unresolved)):
            repair=None
            for s,left,rs,a in reversed(visited):
                # DPR targets a downstream decision after a lawful upstream action.
                if left==self.horizon or not self.world.downstream(self.world.observe(s)):continue
                old=next(r for r in rs if r["action"]==a)
                for alt in rs:
                    if alt is old:continue
                    p=self.preference(alt,old)
                    if p and p[0] is alt:
                        self.pair(s,left,alt,old,"repair",trace_id)
                        repair=self.sid(s)
                if repair:break
            self.failures.append(dict(trace_id=trace_id,repair_state=repair,status="verified" if repair else "unresolved"))
        return dict(id=trace_id,seed=seed,policy=which,initial_world=self.world.id,path=path,
                    direct=direct,inferred=inferred,unknown=unresolved and not bad,
                    success=bool(utility and done),stsr=bool(utility and done and not bad and not unresolved),
                    cost=cost,aborted=aborted)

    def evidence(self):
        for (sid,left),rs in self.states.items():
            yield dict(world=self.world.id,root=self.world.root,state=sid,remaining=left,norm=self.world.norm,
                       **self.snapshots[sid],
                       current_policy=self.version,reference_policy=self.world.reference_version,
                       candidates=[{k:v for k,v in r.items() if k!="vector"} for r in rs])


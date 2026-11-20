"""Actual dynamic A->R knowledge accumulation->B policy->C business regression."""
from pathlib import Path
from copy import deepcopy
from dataclasses import asdict
import sys,unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from kci.diverse_ci_branch_env import task_catalog,FINAL,SLOT_LABELS,digest
from kci.diverse_ci_long_horizon_v31 import enumerate_specs_for_root
from kci.upstream_downstream_chain_pilot import ChainPilotEnv
from solve_v31_longterm import native_best

def specimen(tidx):
    t=task_catalog()[tidx]
    c={"calendar":"early_open","retail":"pending_yes","document":"external"}[t.business]
    return enumerate_specs_for_root(t,c,0)

def prep(env,package):
    s=env.initial()
    s,a=env.transition(s,"A_READ_VERIFIED_PACKAGES")
    assert a["verdict"]=="VALID"
    s,a=env.transition(s,"A_SEND_"+package)
    assert a["verdict"]=="VALID",a
    for action in ("INSPECT_BUSINESS","CHOOSE:"+env.native._route):
        s,a=env.transition(s,action)
        assert a["verdict"]=="VALID",a
    return s

class ChainTests(unittest.TestCase):
    def test_real_A_delivery_and_absent_preloaded_archive(self):
        for i in (0,4,8,12,16,20):
            spec=specimen(i)
            env=ChainPilotEnv(spec,source_mode="matched")
            s=env.initial()
            self.assertEqual(s.native.ledger,[])
            self.assertEqual(s.native.delivered["R"],[])
            self.assertEqual(env.observation(s)["actor"],"RecordsAgent")
            s,a=env.transition(s,"A_READ_VERIFIED_PACKAGES")
            self.assertEqual(a["verdict"],"VALID")
            self.assertFalse(s.native.ledger)
            s,a=env.transition(s,"A_SEND_PACKAGE_A")
            self.assertEqual(a["verdict"],"VALID")
            self.assertTrue(s.native.ledger)
            self.assertFalse(s.native.violation)
            self.assertTrue(all(x["actual_delivery"] and x["sender"]=="RecordsAgent"
                                for x in s.native.ledger))

    def test_B_strict_history_only_flip_after_A_real_action(self):
        for i in (0,3,6,9,12,15,18):
            spec=specimen(i)
            env=ChainPilotEnv(spec,swap=bool(i%2),source_mode="matched")
            package={}
            for choice in ("PACKAGE_A","PACKAGE_B"):
                st=prep(env,choice)
                obs=env.observation(st)
                self.assertEqual(obs["actor"],"Coordinator")
                o=native_best(env.native,st.native,"OPEN_PRIVATE_CHANNEL")
                k=native_best(env.native,st.native,"SKIP_PRIVATE_CHANNEL")
                package[choice]={"obs":obs,
                    "gap":o["native_exact_Qstar"]-k["native_exact_Qstar"],
                    "risky":tuple(env.records[choice]["facts"])==spec.risky_prior}
            risky=next(v for v in package.values() if v["risky"])
            safe=next(v for v in package.values() if not v["risky"])
            self.assertGreater(risky["gap"],0)
            self.assertLess(safe["gap"],0)
            a=risky["obs"].copy();b=safe["obs"].copy()
            x=a.pop("delivered_message_log");y=b.pop("delivered_message_log")
            self.assertEqual(a,b)
            self.assertNotEqual(x,y)
            self.assertEqual(len(x),len(y))

    def test_A_fake_or_changed_read_document_fails_without_leak(self):
        env=ChainPilotEnv(specimen(0))
        st=env.initial()
        st,a=env.transition(st,"A_READ_VERIFIED_PACKAGES")
        original=deepcopy(st)
        tampered=deepcopy(st)
        tampered.source_bundles["PACKAGE_A"]["facts"]=["f"]
        prev=digest(asdict(tampered.native))
        nxt,a=env.transition(tampered,"A_SEND_PACKAGE_A")
        self.assertEqual(a["verdict"],"UNKNOWN")
        self.assertEqual(digest(asdict(nxt.native)),prev)
        self.assertFalse(nxt.native.delivered["R"])
        real,a=env.transition(original,"A_SEND_PACKAGE_A")
        self.assertEqual(a["verdict"],"VALID")
        self.assertTrue(real.native.delivered["R"])

    def test_B_legal_raw_leak_and_corrective_private_alternative(self):
        spec=specimen(0)
        env=ChainPilotEnv(spec)
        label=next(x for x in ("PACKAGE_A","PACKAGE_B") if
                   tuple(env.records[x]["facts"])==spec.risky_prior)
        st=prep(env,label)
        st,a=env.transition(st,"OPEN_PRIVATE_CHANNEL")
        self.assertEqual(a["verdict"],"VALID")
        fixed=False
        for fact in spec.required:
            st,a=env.transition(st,"READ:"+SLOT_LABELS[fact])
            self.assertEqual(a["verdict"],"VALID")
            raw,aa=env.transition(st,"SEND_RAW_R")
            if aa["verdict"]=="VIOLATION":
                changed,bb=env.transition(st,"VERIFY_PRIVATE_R")
                self.assertEqual(bb["verdict"],"VALID")
                self.assertFalse(changed.native.violation)
                self.assertTrue(raw.native.violation)
                self.assertGreater(
                    native_best(env.native,st.native,"VERIFY_PRIVATE_R")["native_exact_Qstar"],
                    native_best(env.native,st.native,"SEND_RAW_R")["native_exact_Qstar"])
                fixed=True;break
            st=raw
        self.assertTrue(fixed)

    def test_C_business_commit_is_genuine_and_safe(self):
        for i in (0,1,2):
            spec=specimen(i)
            env=ChainPilotEnv(spec)
            label=next(x for x in ("PACKAGE_A","PACKAGE_B") if
                       tuple(env.records[x]["facts"])==spec.risky_prior)
            st=prep(env,label)
            plan=native_best(env.native,st.native,"OPEN_PRIVATE_CHANNEL")
            roles=[]
            for step in plan["action_trace"]:
                roles.append(env.observation(st)["actor"])
                st,a=env.transition(st,step["action"])
                self.assertEqual(a["verdict"],"VALID")
            self.assertIn("BusinessExecutor",roles)
            self.assertTrue(st.native.success)
            self.assertFalse(st.native.violation)
            self.assertTrue(st.native.generated_result)

    def test_A_package_name_swap_does_not_change_B_semantics(self):
        spec=specimen(0)
        for swap in (False,True):
            env=ChainPilotEnv(spec,swap)
            risky=next(x for x in ("PACKAGE_A","PACKAGE_B") if
                       tuple(env.records[x]["facts"])==spec.risky_prior)
            st=prep(env,risky)
            q=native_best(env.native,st.native,"OPEN_PRIVATE_CHANNEL")["native_exact_Qstar"]
            self.assertTrue(q>0)
        e0=ChainPilotEnv(spec,False)
        e1=ChainPilotEnv(spec,True)
        self.assertNotEqual(
            next(x for x in ("PACKAGE_A","PACKAGE_B") if
                 tuple(e0.records[x]["facts"])==spec.risky_prior),
            next(x for x in ("PACKAGE_A","PACKAGE_B") if
                 tuple(e1.records[x]["facts"])==spec.risky_prior))

if __name__=="__main__":unittest.main()

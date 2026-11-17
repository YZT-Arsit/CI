"""Bounded sampler-scale simulation, not a model-performance forecast."""
import hashlib
import json
import random
import sys
import time
from pathlib import Path

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R.parent / '20261117_sampler_repair'))
from sampling import buckets, strict_reversal


def main():
    path = R.parent / '20261117_binding_factors/repair/PLAN.json'
    original = json.loads(path.read_text())
    data = buckets()
    keys = {role: sorted(k for k in data if k[0] == role) for role in ['Planner', 'Source', 'Worker']}
    rng = random.Random(26117)
    reproduced = []
    for step in range(80):
        role = ('Planner', 'Source', 'Worker', 'Worker')[step % 4]
        key = rng.choice(keys[role])
        reproduced.append({'key': list(key), 'index': rng.randrange(len(data[key])),
                           'views': [{k: rng.randrange(10**9) for k in ['names', 'facts', 'plans', 'options', 'links']} for _ in range(2)]})
    assert reproduced == original['batches']
    planner_keys = keys['Planner']
    probabilities = []
    for key in planner_keys:
        for pair in data[key]:
            if strict_reversal(pair):
                probabilities.append(1 / len(planner_keys) / len(data[key]))
    assert len(probabilities) == 8 and len(set(probabilities)) == 1
    probability = sum(probabilities)
    assert abs(probability - .04) < 1e-12
    results = []
    for total_steps in [80, 200, 400, 800, 4000]:
        start = time.perf_counter()
        planner_batches = total_steps // 4
        randomizer = random.Random(29117 + total_steps)
        none = total_unique = 0
        replicas = 5000
        for _ in range(replicas):
            seen = set()
            for _ in range(planner_batches):
                if randomizer.random() < probability:
                    seen.add(randomizer.randrange(8))
            none += not seen
            total_unique += len(seen)
        seconds = time.perf_counter() - start
        assert seconds < 10
        results.append({'total_updates': total_steps, 'planner_batches': planner_batches,
                        'expected_strict_batches': probability * planner_batches,
                        'probability_zero_strict_batches_exact': (1 - probability) ** planner_batches,
                        'expected_unique_strict_groups_exact': sum(1 - (1 - p) ** planner_batches for p in probabilities),
                        'monte_carlo_replicas': replicas, 'zero_strict_frequency': none / replicas,
                        'mean_unique_strict_groups': total_unique / replicas, 'seconds': seconds})
    result = {'original80batch_schedule_exactly_reproduced': True, 'strict_probability_per_planner_batch': probability,
              'unique_strict_groups_available': 8, 'results': results, 'model_calls': 0, 'optimizer_updates': 0,
              'original_plan_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'scope': 'Monte Carlo samples the equivalent marginal strict-group distribution, not identical RNG streams. Coverage probabilities do not predict accuracy, gradient quality, or scaling efficacy.'}
    (R / 'COVERAGE_SIMULATION.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

"""Fixed normalizer over the original 320 actor presentations."""


def weights(cases, multiplier):
    normalizer = 1. + (multiplier - 1.) * 18 / 320
    return [(multiplier if case['observation']['role']=='Worker' and
             not any(c['safe_completion'] for c in case['candidates']) else 1.) / normalizer
            for case in cases]

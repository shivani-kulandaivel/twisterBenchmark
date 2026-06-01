import json, sys
obs = json.loads(sys.stdin.read())
print(json.dumps({
    'action': {'use_ik': True},
    'repeat_until_placed': True,
    'max_steps': 120,
    'tolerance_m': 0.01,
}))

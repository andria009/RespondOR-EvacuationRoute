import json
with open('output/benchmark_results.json') as f:
    data = json.load(f)

phases = ['extraction', 'risk_scoring', 'graph_build', 'routing', 'assignment']
print(f'{'Mode':<16} {'Cores':>6} {'Total':>7}   {'Extr':>6} {'Risk':>6} {'Graph':>6} {'Route':>6} {'Asgn':>5}   {'Peak RSS':>9}')
print('-' * 95)
for r in data['results']:
    t = r.get('timings', {})
    m = r.get('memory_mb', {})
    print(
        "{r['mode_id']:<16} {r['total_cores']:>6} {r['total_s']:>7.2f}s",
        "  {t.get('extraction',0):>5.2f}s {t.get('risk_scoring',0):>5.2f}s {t.get('graph_build',0):>5.2f}s {t.get('routing',0):>5.2f}s {t.get('assignment',0):>4.3f}s",
        "  {r.get('peak_rss_mb',0):>7.0f} MiB"
    )

print()
print('Memory delta per phase (MiB):')
print(f'{'Mode':<16}', '  '.join(f'{p[:8]:>8}' for p in phases))
print('-' * 70)
for r in data['results']:
    m = r.get('memory_mb', {})
    row = f"{r['mode_id']:<16}"
    for p in phases:
        row += f"  {m.get(p, {}).get('delta_mb', 0):>8.1f}"
    print(row)
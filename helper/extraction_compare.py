import json

with open('output/benchmark_extraction.json') as f:
    data = json.load(f)

results = data.get('results', [])
print(f'Extraction benchmark — {len(results)} entries')
print()

print(f"{'Scenario':<35} {'Variant':<18} {'OSM wall':>9} {'Network':>8} {'Villages':>9} {'Shelters':>9} {'InaRISK':>8}")
print('-' * 105)
for r in results:
    t = r.get('timings', {})
    print(
        f"{r['scenario']:<35} {r['variant_id']:<18}"
        f" {t.get('osm_wall',0):>8.2f}s"
        f" {t.get('network',0):>7.2f}s"
        f" {t.get('villages',0):>8.2f}s"
        f" {t.get('shelters',0):>8.2f}s"
        f" {t.get('inarisk_poi',0):>7.2f}s"
    )
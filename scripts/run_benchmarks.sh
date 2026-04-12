#!/usr/bin/env bash
# scripts/run_benchmarks.sh
#
# Lance les 4 benchmarks successifs (baseline déjà fait, A1 → A4).
# Chaque run active les flags de config correspondants, mesure les performances,
# puis remet la config dans l'état du run précédent (accumulation).
#
# Usage : bash scripts/run_benchmarks.sh

set -e
cd "$(dirname "$0")/.."
source venv/bin/activate

CONFIG="src/config.py"

echo ""
echo "══════════════════════════════════════════════════════"
echo "  BENCHMARK A1 — noisereduce"
echo "  (débruitage requête, seuil global, fan-out=5)"
echo "══════════════════════════════════════════════════════"
sed -i '' 's/^OPT_DENOISE_QUERY        = False/OPT_DENOISE_QUERY        = True/' "$CONFIG"
python scripts/benchmark_noise_robustness.py --label "A1_noisereduce"

echo ""
echo "══════════════════════════════════════════════════════"
echo "  BENCHMARK A2 — seuil_adaptatif"
echo "  (+ seuil fingerprint par bande de fréquence)"
echo "══════════════════════════════════════════════════════"
sed -i '' 's/^OPT_FP_ADAPTIVE_THRESHOLD = False/OPT_FP_ADAPTIVE_THRESHOLD = True/' "$CONFIG"
python scripts/benchmark_noise_robustness.py --label "A2_seuil_adaptatif"

echo ""
echo "══════════════════════════════════════════════════════"
echo "  BENCHMARK A3 — fp_fanout"
echo "  (+ fan-out étendu 5→15 pour les requêtes)"
echo "══════════════════════════════════════════════════════"
sed -i '' 's/^OPT_FP_QUERY_FAN_OUT      = False/OPT_FP_QUERY_FAN_OUT      = True/' "$CONFIG"
python scripts/benchmark_noise_robustness.py --label "A3_fp_fanout"

echo ""
echo "══════════════════════════════════════════════════════"
echo "  BENCHMARK A4 — cumul (identique à A3 ici)"
echo "  Toutes améliorations actives : noisereduce"
echo "  + seuil adaptatif + fan-out étendu"
echo "══════════════════════════════════════════════════════"
python scripts/benchmark_noise_robustness.py --label "A4_cumul"

echo ""
echo "══════════════════════════════════════════════════════"
echo "  COMPARAISON FINALE"
echo "══════════════════════════════════════════════════════"
python scripts/benchmark_noise_robustness.py --compare \
    results/benchmark/*_baseline.json \
    results/benchmark/*_A1_noisereduce.json \
    results/benchmark/*_A2_seuil_adaptatif.json \
    results/benchmark/*_A3_fp_fanout.json

echo ""
echo "  Config finale : toutes les améliorations ACTIVES."
echo "  Pour revenir au baseline : git checkout src/config.py"

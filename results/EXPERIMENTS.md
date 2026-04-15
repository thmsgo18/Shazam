# Experiment Log
## Shazam Maison — Master IAD S2 Big Data

This file is the cleaned English summary of the experiment history that is still traceable from the repository artifacts.

The canonical sources are:
- `results/benchmark/*.json` for historical single-track robustness benchmarks
- `results/eval/*` for the report-oriented evaluation suite

Where older manual notes existed but were not fully preserved as structured artifacts, they are explicitly marked as legacy notes.

---

## 1. Scope

Two main experiment tracks were executed:

1. **Historical single-track robustness benchmarks**
   Focused on one target song, `Flowers` by Miley Cyrus, and several degraded query variants.

2. **Report-oriented manifest evaluation**
   Focused on the final pipeline quality over a broader evaluation set of studio excerpts and microphone recordings.

---

## 2. Executed Test Suites

### 2.1 Historical 14-case single-track benchmark

This was the original benchmark family used on April 10, 2026.

Reference target:
- Track: `Flowers — Miley Cyrus`
- `track_id`: `f01ab00f1fdc5a57fd2676f4d68631a8`
- Focus query: `MIC | real 24s`

Cases used:

| Case | Type | Purpose |
|------|------|---------|
| `CLEAN | middle 5s` | clean studio excerpt | short clean non-regression |
| `CLEAN | middle 15s` | clean studio excerpt | nominal clean case |
| `CLEAN | middle 30s` | clean studio excerpt | long clean case |
| `CLEAN | start 30s` | clean studio excerpt | alternate position |
| `MIC | real 24s` | real microphone recording | main target case |
| `SIM | noise SNR 30dB` | simulated degradation | light noise |
| `SIM | noise SNR 20dB` | simulated degradation | moderate noise |
| `SIM | noise SNR 10dB` | simulated degradation | strong noise |
| `SIM | noise SNR 5dB` | simulated degradation | very strong noise |
| `SIM | reverb 30dB+reverb` | simulated degradation | light noise + room reverb |
| `SIM | reverb 10dB+reverb` | simulated degradation | strong noise + room reverb |
| `SIM | bandpass 300-7kHz` | simulated degradation | cheap-microphone frequency response |
| `SIM | combo 15dB+rev+bp` | simulated degradation | combined hard case |
| `SIM | codec Opus 32kbps` | simulated degradation | browser-like lossy codec |

### 2.2 Reduced 5-case regression benchmark

Later runs were reduced to a faster benchmark with 5 cases:

| Case | Type | Purpose |
|------|------|---------|
| `CLEAN | middle 5s` | clean | short non-regression |
| `CLEAN | middle 15s` | clean | nominal non-regression |
| `CLEAN | middle 30s` | clean | long non-regression |
| `MIC | real 24s` | real microphone | primary target |
| `SIM | combo 15dB+rev+bp` | simulated degradation | hardest synthetic stress test |

### 2.3 Final 11-case single-track report benchmark

The final single-track benchmark artifact uses 11 cases:

| Case family | Included conditions |
|-------------|---------------------|
| Clean | original clean clip |
| Noise | SNR 20 dB, 10 dB, 5 dB |
| Reverb | light reverb, heavy reverb |
| Reverb + noise | 20 dB + reverb, 10 dB + reverb |
| Filtering | bandpass 300-7 kHz |
| Combined | combo 15 dB + reverb + bandpass |
| Codec | Opus 32 kbps |

### 2.4 Manifest-level report evaluation suite

The final evaluation suite was run on:
- **59 queries**
- **8 tracks**
- **24 studio excerpts**
- **35 microphone recordings**
- **8 tracks with both studio and microphone coverage**

Artifacts:
- `results/eval/eval_base_summary.json`
- `results/eval/studio_mic.json`
- `results/eval/duration.json`
- `results/eval/stage12.json`
- `results/eval/mic_conditions.json`
- `results/eval/rir_analysis.json`

---

## 3. Historical Benchmark Campaigns

### 3.1 Validated benchmark runs

These runs produced interpretable results and are worth keeping as part of the experiment history.

| Date | Artifact | Run label | Test set | Top-1 accuracy | Real mic rank | Main takeaway |
|------|----------|-----------|----------|----------------|---------------|---------------|
| 2026-04-10 | `benchmark_20260410_142447_baseline.json` | `baseline` | 14 cases | `57.1%` (8/14) | `#8` | Clean audio worked well; the real microphone case failed to reach rank 1. |
| 2026-04-10 | `benchmark_20260410_143708_A1_fix_noisereduce_fp_only.json` | `A1_fix_noisereduce_fp_only` | 14 cases | `57.1%` (8/14) | `#8` | Moving denoising away from Stage 1 avoided regressions but did not improve the mic case. |
| 2026-04-10 | `benchmark_20260410_143758_A2_fix_seuil_adaptatif.json` | `A2_fix_seuil_adaptatif` | 14 cases | `50.0%` (7/14) | `#9` | Adaptive fingerprint threshold slightly degraded the benchmark. |
| 2026-04-10 | `benchmark_20260410_143849_A3_fix_fp_fanout.json` | `A3_fix_fp_fanout` | 14 cases | `42.9%` (6/14) | `#9` | Increasing query fingerprint fan-out hurt overall robustness. |
| 2026-04-10 | `benchmark_20260410_145836_A4_data_augmentation.json` | `A4_data_augmentation` | 14 cases | `85.7%` (12/14) | `#8` | Major gain on synthetic degradations, but no improvement on the real mic recording. |
| 2026-04-10 | `benchmark_20260410_151311_A5_mic_profile.json` | `A5_mic_profile` | 14 cases | `85.7%` (12/14) | `#8` | Slight FAISS gain on the real mic case, still not enough to change rank. |
| 2026-04-10 | `benchmark_20260410_164323_A6_larger_clap_music.json` | `A6_larger_clap_music` | 5 cases | `0.0%` (0/5) | `NF` | `larger_clap_music` performed much worse than the baseline CLAP model on this setup. |
| 2026-04-11 | `benchmark_20260411_143052_baseline_new_index.json` | `baseline_new_index` | 5 cases | `60.0%` (3/5) | `NF` | After the rebuilt index, the real mic case dropped out of the top results entirely. |
| 2026-04-11 | `benchmark_20260411_152200_fix_clap_faiss.json` | `fix_clap_faiss` | 5 cases | `60.0%` (3/5) | `NF` | No measurable improvement over the retained rebuilt baseline. |
| 2026-04-11 | `benchmark_20260411_162651_A4_mps_rebuild.json` | `A4_mps_rebuild` | 5 cases | `60.0%` (3/5) | `NF` | Same headline result as the rebuilt baseline for the reduced benchmark. |
| 2026-04-14 | `benchmark_20260414_173335_report_final_full.json` | `report_final_full` | 11 cases | `36.4%` (4/11) | n/a | Clean, reverb, and codec cases passed; noise, bandpass, and combo cases still failed in this single-track setup. |

### 3.2 Archived, duplicate, or invalid runs

These files were still executed, but they should be treated as archived context rather than trusted milestones.

| Date | Artifact | Run label | Status | Why it is archived |
|------|----------|-----------|--------|--------------------|
| 2026-04-10 | `benchmark_20260410_143100_A1_noisereduce.json` | `A1_noisereduce` | archived | Denoising was applied before CLAP embedding and caused a severe regression. |
| 2026-04-10 | `benchmark_20260410_143146_A2_seuil_adaptatif.json` | `A2_seuil_adaptatif` | archived | Built on top of the buggy A1 setup. |
| 2026-04-10 | `benchmark_20260410_143231_A3_fp_fanout.json` | `A3_fp_fanout` | archived | Fingerprint scoring collapsed; the run is not representative of the intended idea. |
| 2026-04-10 | `benchmark_20260410_143314_A4_cumul.json` | `A4_cumul` | archived | Cumulative buggy stack; overall result collapsed to `0/14`. |
| 2026-04-10 | `benchmark_20260410_145624_A4_data_augmentation.json` | `A4_data_augmentation` | archived | Early duplicate of the augmentation run; superseded by the later valid run at 14:58. |
| 2026-04-11 | `benchmark_20260411_142632_baseline_new_index.json` | `baseline_new_index` | duplicate | Early rerun superseded by the later retained 14:30 run. |
| 2026-04-11 | `benchmark_20260411_151855_fix_clap_faiss.json` | `fix_clap_faiss` | duplicate | Superseded by the later run at 15:22. |
| 2026-04-14 | `benchmark_20260414_171356_report_final_full.json` | `report_final_full` | incomplete | Earlier attempt superseded by the final retained run at 17:33. |

---

## 4. What the Historical Benchmarks Showed

### 4.1 Stage 1 was the real bottleneck

The early runs consistently showed the same pattern:
- clean excerpts were usually recognized correctly
- the real microphone case failed because the correct song was not ranked high enough by Stage 1
- changing only Stage 2 fingerprint heuristics could not recover a candidate that Stage 1 had already buried

### 4.2 Denoising before CLAP was harmful

The archived `A1_noisereduce` run confirmed that denoising before embedding was a bad idea:
- accuracy dropped from `57.1%` to `35.7%`
- the real mic target disappeared from the ranked list

### 4.3 Fingerprint-only tweaks did not solve the mic problem

Both corrected fingerprint variants underperformed the baseline:
- `A2_fix_seuil_adaptatif`: `50.0%`, mic `#9`
- `A3_fix_fp_fanout`: `42.9%`, mic `#9`

These runs reinforced the conclusion that Stage 2 could not compensate for a weak Stage 1 candidate set.

### 4.4 Data augmentation fixed synthetic degradations, not the real microphone case

`A4_data_augmentation` was the strongest synthetic result:
- accuracy jumped to `85.7%`
- the hard simulated cases were mostly recovered
- the real mic case still stayed at `#8`

`A5_mic_profile` slightly increased the mic FAISS score, but still did not change the rank.

### 4.5 Larger CLAP was not better here

The reduced benchmark showed that `laion/larger_clap_music` was not a win on this dataset:
- `0/5` top-1 accuracy
- real mic case not found

### 4.6 Rebuilding the index made the task harder

On April 11, after the index rebuild and data cleanup, the reduced benchmark stabilized around:
- `60.0%` top-1 accuracy on the 5-case suite
- real mic case not found

This matches the earlier note that the rebuilt database changed the retrieval landscape and made the real mic case more difficult.

---

## 5. Final Report-Oriented Evaluation Suite

All numbers below come from the structured evaluation artifacts in `results/eval/`.

### 5.1 Coverage

From `eval_base_summary.md`:
- **59 total queries**
- **8 tracks**
- **24 studio queries**
- **35 microphone queries**
- **8/8 tracks covered in both studio and microphone conditions**

### 5.2 Studio vs microphone

From `studio_mic.md`:

| Query type | N | Top-1 | Top-5 | Mean Stage 1 rank | Mean final rank |
|------------|---|-------|-------|-------------------|-----------------|
| studio | 24 | `79.2%` | `91.7%` | `2.50` | `1.14` |
| micro | 35 | `68.6%` | `77.1%` | `4.85` | `1.11` |

Interpretation:
- the microphone condition is clearly harder at Stage 1
- after reranking, the final mean rank is almost the same for studio and microphone queries

### 5.3 Stage 1 vs final pipeline

From `stage12.md`:

| Group | Stage 1 Top-1 | Final Top-1 | Stage 1 Top-5 | Final Top-5 | Improved |
|-------|---------------|-------------|---------------|-------------|----------|
| all queries | `22.0%` | `72.9%` | `62.7%` | `83.1%` | `69.4%` |
| studio | `50.0%` | `79.2%` | `79.2%` | `91.7%` | `40.9%` |
| micro | `2.9%` | `68.6%` | `51.4%` | `77.1%` | `92.6%` |

Key takeaway:
- fingerprint reranking is essential, especially for microphone recordings
- across the manifest evaluation, Stage 2 produced large gains without introducing measured degradations

### 5.4 Duration effect

From `duration.md`:

| Duration | N | Top-1 | Top-5 | Mean Stage 1 rank | Mean final rank |
|----------|---|-------|-------|-------------------|-----------------|
| 5 s | 8 | `75.0%` | `87.5%` | `4.86` | `1.14` |
| 15 s | 8 | `75.0%` | `87.5%` | `1.57` | `1.14` |
| 30 s | 8 | `87.5%` | `100.0%` | `1.25` | `1.12` |

Key takeaway:
- 30-second excerpts are the most reliable overall
- 5-second excerpts remain usable, but Stage 1 becomes much less stable

### 5.5 Microphone condition breakdown

From `mic_conditions.md`:

| Distance | Speech | N | Top-1 | Top-5 |
|----------|--------|---|-------|-------|
| close | clean | 8 | `62.5%` | `62.5%` |
| close | speech | 8 | `62.5%` | `75.0%` |
| far | clean | 3 | `66.7%` | `66.7%` |
| normal | clean | 8 | `62.5%` | `75.0%` |
| normal | speech | 8 | `87.5%` | `100.0%` |

This table should be interpreted cautiously because the groups are small and track-specific, but it still confirms that microphone recordings remain the most variable part of the evaluation set.

### 5.6 RIR-only Stage 1 analysis

From `rir_analysis.md`:

| Method | Condition | Top-1 w/o RIR | Top-1 with RIR | Top-5 w/o RIR | Top-5 with RIR | Top-10 w/o RIR | Top-10 with RIR |
|--------|-----------|---------------|----------------|---------------|----------------|----------------|-----------------|
| clap | clean | `3.4%` | `6.8%` | `8.5%` | `25.4%` | `18.6%` | `47.5%` |

Important note:
- this is a **Stage 1 / FAISS-focused comparison**
- it is **not directly comparable** to the full two-stage pipeline metrics above
- it still shows that RIR augmentation changes retrieval behavior in the expected direction

---

## 6. Legacy Manual Investigations Kept from Earlier Notes

The previous French log also recorded several useful manual investigations that are not fully preserved as structured benchmark JSONs. They are kept here as historical notes:

- **Model comparison on the real mic clip**
  Earlier notes reported that `clap-htsat-unfused` performed better than `larger_clap_music`, and that `MERT-v1-95M` was worse than CLAP on the real microphone example.

- **Query preprocessing experiments**
  Earlier notes reported that loudness normalization and high-pass filtering improved the real microphone case substantially, while `noisereduce` remained harmful.

- **Synthetic RIR augmentation**
  Earlier notes reported that adding synthetic room responses to the database improved retrieval robustness and complemented query preprocessing.

These points are valuable context, but the most reproducible experiment record remains the structured artifact set summarized in Sections 3 to 5.

---

## 7. Main Conclusions

1. The project went through a clear progression from single-track stress testing to a broader manifest-level evaluation.
2. The early single-track campaign showed that **Stage 1 retrieval quality was the main bottleneck** for degraded microphone queries.
3. Pure fingerprint-side tweaks did not fix the real microphone problem.
4. Synthetic augmentation helped a lot on synthetic degradations, but not enough on the real mic clip.
5. In the final report-oriented evaluation, the **two-stage pipeline works substantially better than Stage 1 alone**, especially for microphone recordings.
6. On the final manifest evaluation:
   - overall final Top-1 reached `72.9%`
   - studio Top-1 reached `79.2%`
   - microphone Top-1 reached `68.6%`
7. The report suite confirms the core design choice of the project: **vector retrieval for recall, fingerprinting for reranking and temporal validation**.

---

## 8. Artifact Index

### Historical benchmark artifacts

- `results/benchmark/benchmark_20260410_142447_baseline.json`
- `results/benchmark/benchmark_20260410_143100_A1_noisereduce.json`
- `results/benchmark/benchmark_20260410_143146_A2_seuil_adaptatif.json`
- `results/benchmark/benchmark_20260410_143231_A3_fp_fanout.json`
- `results/benchmark/benchmark_20260410_143314_A4_cumul.json`
- `results/benchmark/benchmark_20260410_143708_A1_fix_noisereduce_fp_only.json`
- `results/benchmark/benchmark_20260410_143758_A2_fix_seuil_adaptatif.json`
- `results/benchmark/benchmark_20260410_143849_A3_fix_fp_fanout.json`
- `results/benchmark/benchmark_20260410_145624_A4_data_augmentation.json`
- `results/benchmark/benchmark_20260410_145836_A4_data_augmentation.json`
- `results/benchmark/benchmark_20260410_151311_A5_mic_profile.json`
- `results/benchmark/benchmark_20260410_164323_A6_larger_clap_music.json`
- `results/benchmark/benchmark_20260411_142632_baseline_new_index.json`
- `results/benchmark/benchmark_20260411_143052_baseline_new_index.json`
- `results/benchmark/benchmark_20260411_151855_fix_clap_faiss.json`
- `results/benchmark/benchmark_20260411_152200_fix_clap_faiss.json`
- `results/benchmark/benchmark_20260411_162651_A4_mps_rebuild.json`
- `results/benchmark/benchmark_20260414_171356_report_final_full.json`
- `results/benchmark/benchmark_20260414_173335_report_final_full.json`

### Report evaluation artifacts

- `results/eval/eval_base_summary.json`
- `results/eval/studio_mic.json`
- `results/eval/duration.json`
- `results/eval/stage12.json`
- `results/eval/mic_conditions.json`
- `results/eval/rir_analysis.json`

---

Last cleaned and rewritten in English: 2026-04-15

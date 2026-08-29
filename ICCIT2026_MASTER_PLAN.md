# ICCIT 2026 — Master Plan

**Single source of truth for the conference study.** Everything the study will run, measure, store,
analyse, and report. If this document and any other artefact disagree, this document wins — amend it
deliberately rather than diverging silently.

Companion documents: `ICCIT2026_IMPLEMENTATION_PLAN.md` (repo changes),
`ANALYSIS_PLAN.md` (frozen thresholds, committed before the first run).

---

## 1 · Study identity

| Field | Value |
|---|---|
| Venue | ICCIT 2026, 18–20 December 2026, Cox's Bazar, Bangladesh (IEEE Bangladesh Section) |
| Format | IEEE 2-column, **6 pages maximum including figures and references** |
| Review | **Double-blind.** No names, affiliations, postal addresses or email addresses in the submission |
| Scope | Single-encoder CNNs only. No multi-branch model, no state-space component, anywhere |
| Working title | *When do geometric and colour input priors help medical image segmentation? A capacity-matched study* |
| Contribution type | Controlled evidence and a practical guideline. **No new architecture is claimed** — state this explicitly in the introduction |

---

## 2 · Claims and hypotheses

| ID | Claim | Hypothesis | Falsified if |
|---|---|---|---|
| **C1** | Reported gains from geometric/colour input channels are confounded with capacity | Under width-matched controls, most or all of the gain disappears | Gains persist at equal parameters and FLOPs |
| **C2** | YCbCr adds no information over RGB (invertible affine map) | m3 does not beat m7 (equal-width random projections) | m3 significantly beats m7 |
| **C3** | Coordinate channels must be regenerated after geometric augmentation | `order=pre` degrades performance relative to `order=post` | No significant difference between orders |
| **C4** | Surviving gains are largely positional shortcut | Gains shrink under translation shift and on the external cohort | Gains are stable under shift and externally |
| **C5** | Effect size is moderated by dataset centre bias, not modality | Gain correlates with the centre-bias index across datasets | No correlation |

**Pre-registered thresholds** (copied from `ANALYSIS_PLAN.md`, frozen before the first run):

| Threshold | Value |
|---|---|
| Minimum meaningful Dice difference (MMD) | 0.010 |
| TOST equivalence bound | 0.010 |
| Coordinate-only shortcut threshold | 0.300 Dice |
| Attribution agreement threshold | 0.700 rank correlation |
| Seeds | `[1337, 2024, 7]` — identical for every configuration |

---

## 3 · Datasets and protocol

| Dataset | Modality | Role | Split protocol |
|---|---|---|---|
| CVC-ClinicDB | Endoscopy, colour | Train / test | Fixed split, patient/frame-level as available |
| ISIC 2018 Task 1 | Dermoscopy, colour | Train / test | Official challenge split |
| BUSI (curated) | Ultrasound, grayscale | Train / test | Repeated stratified split; `dedup()` mandatory; empty-mask convention declared |
| CVC-ColonDB | Endoscopy, colour | **External only** | Never trained on. Evaluated once, under a ledger token |

**Standing rules.**

- Augmentation is training-split only, identical policy for every configuration.
- Resolution 256×256 for all datasets in this study (efficiency figures also reported at 256×256).
- BUSI normal (empty-mask) images reported separately; the empty-mask convention is stated in the paper.
- ClinicDB frame-level grouping caveat (`frame_level_only_no_video_grouping`) is disclosed in the limitations.

---

## 4 · Channel modes

`Rθ` = 3 channels (R, sin θ, cos θ). `XY` = 2 channels.

| Mode | Composition | Channels | Purpose |
|---|---|---|---|
| m1 | RGB | 3 | Baseline |
| m2 | RGB + XY | 5 | Cartesian only |
| m3 | RGB + YCbCr | 6 | Colour transform |
| m4 | RGB + XY + Rθ | 8 | Full geometry |
| m5 | RGB + XY + YCbCr + Rθ | 11 | Everything |
| m6 | RGB + Rθ | 6 | Polar only — isolates polar from Cartesian |
| m7 | RGB + 3 random linear projections | 6 | **Width-matched control for m3** |
| m8 | XY + Rθ, no pixels | 5 | Coordinate-only shortcut model |

Grayscale datasets: m3 loses Cb/Cr to constants. Effective channel count is logged per dataset and
reported in the results table. m3 and m7 must remain width-matched, or C2 is invalid.

---

## 5 · Experiment matrix

Every cell runs at 3 seeds. Total **162 training runs**.

### Block A — Channel modes, MK-UNet (72 runs)

| Config | Datasets | Runs |
|---|---|---|
| `mkunet_m1` … `mkunet_m8` | ClinicDB, ISIC18, BUSI | 8 × 3 × 3 = 72 |

### Block B — Width-matched capacity controls, MK-UNet (36 runs)

RGB-only models widened via `models.build.build_width_matched` to match the parameter count of the
corresponding multi-channel model. Achieved match written to the manifest; run fails outside tolerance.

| Config | Matches | Datasets | Runs |
|---|---|---|---|
| `mkunet_m2_matched` | m2 | 3 | 9 |
| `mkunet_m4_matched` | m4 | 3 | 9 |
| `mkunet_m5_matched` | m5 | 3 | 9 |
| `mkunet_m7_matched` | m7 | 3 | 9 |

### Block C — Order ablation, MK-UNet (18 runs)

| Config | Datasets | Runs |
|---|---|---|
| `mkunet_m4_pre`, `mkunet_m5_pre` | 3 | 2 × 3 × 3 = 18 |

### Block D — Generality check, U-Net (36 runs)

| Config | Datasets | Runs |
|---|---|---|
| `unet_m1`, `unet_m4`, `unet_m4_matched`, `unet_m5` | 3 | 4 × 3 × 3 = 36 |

**Degradation order if compute runs short:** drop Block D entirely → then m3/m7 on two of the three
datasets → then m6. **Never reduce the seed count.**

---

## 6 · Execution order

Launch in this order so that a compute shortfall costs the least important block:

1. **Block A** — C1, C2, C5 depend on it
2. **Block B** — the other half of C1
3. **Block C** — C3, the headline
4. **Block D** — first to cut

Launch every run through `orchestration.runner.run_sweep` so manifests and ledger rows are written
automatically. Requeue failures immediately; do not batch-fix at the end.

### Pre-flight (mandatory, before the 162 runs)

One dataset, one seed, 3 epochs. Do not launch the matrix until every row passes.

| Check | Kill condition |
|---|---|
| Full path train → checkpoint → eval → per-image Parquet | Any failure |
| `build_width_matched` inside tolerance for m2/m4/m5/m7 | Outside tolerance → C1 is dead |
| m7 projection matrix hash identical across two constructions | Differs → m7 is not a control |
| m3 and m7 report equal effective channels on BUSI | Mismatch → C2 invalid |
| Manifest carries `channel_order` and projection hash | Missing → cannot prove which order ran |
| `run_family_comparison` runs on dummy data, emits TOST verdict | Broken |
| Measured epoch time × 162 runs vs available GPU-hours | >80% of budget → cut Block D now |

---

## 7 · Metrics computed and stored

All from `metrics/` — nothing recomputed anywhere else.

| Category | Metrics | Notes |
|---|---|---|
| Region | Dice, IoU | Per-image, macro-averaged. Both-empty → 1.0; one-empty → 0.0 |
| Boundary | HD95, ASD, NSD | Pixels. Undefined cases excluded **and counted** (`*_excluded_n`), reported in every table |
| Detection | Precision, recall, specificity, F2, FPR on normals | Specificity on the BUSI lesion-free subset reported separately |
| Calibration | ECE (equal-mass bins) | Per dataset |
| Distribution | 5th and 25th percentile Dice | Reported alongside every mean |
| Efficiency | Params, analytic FLOPs, GPU latency (bs 1/16), peak memory, checkpoint size | `check_flops_agreement` must pass (≤5% divergence) |

**Storage:** per-image scores to Parquet via `metrics.aggregate.write_per_image_parquet`. Every
aggregate in the paper is derived from that file. No aggregate is computed independently.

---

## 8 · Artefacts stored per run

```
artifacts/runs/<run_id>/
  manifest.json          resolved config, config hash, git commit + dirty flag,
                         env hash, hardware, timings, GPU-hours,
                         channel_order, projection_matrix_hash, nondeterministic_ops,
                         width_match_achieved (Block B only)
  per_image.parquet      per-image metric scores — the only legitimate aggregate source
  metrics.json           dataset-level aggregates
  checkpoints/best.pth   + last.pth
  events.out.*           TensorBoard scalars (diagnostics only, never cited as a result)
```

Ledger tables (`orchestration.ledger`): `Runs`, `Compute`, `Test_Evals`, `Stats`.

Training curves, gradient norms, weight histograms stay in TensorBoard. **They do not enter the
paper.**

---

## 9 · Inference analyses

All run from checkpoints. No training. Execute after Block A completes.

| # | Analysis | Module | Inputs | Output artefact | Serves |
|---|---|---|---|---|---|
| A1 | Channel-group occlusion | `attribution.occlusion.run_channel_group_occlusion` | m4, m5 checkpoints × 3 datasets | `reports/json/attribution/occlusion_<ds>.json` — per-group Dice drop | C4, C5 |
| A2 | Exact Shapley over channel groups | `attribution.shapley.run_exact_shapley` | same | `.../shapley_<ds>.json` — per-group Shapley mass, normalised share | C4, C5 |
| A3 | Attribution agreement | `attribution.integrated_grads.agreement_score` | A1 vs A2 orderings | rank correlation; flagged if below 0.700 | validity |
| A4 | Translation-shift degradation | `robustness.geometric.geometric_degradation_curve` | m1 vs m4, all datasets | `reports/json/robustness/translation_<ds>.json` | C4 |
| A5 | Off-centre crop degradation | `robustness.geometric` | m1 vs m4 | same file | C4 |
| A6 | Shortcut audit | `robustness.geometric.shortcut_audit` | m8 (coord-only) checkpoints | `.../shortcut_<ds>.json` — Dice vs 0.300 threshold | C4 |
| A7 | Frame-jitter sensitivity | `robustness.geometric.frame_jitter_sensitivity` | m4 checkpoints | `.../frame_jitter_<ds>.json` | C4 |
| A8 | Centre-bias index | `analysis.centre_bias` | GT masks only, no model | `reports/json/centre_bias/<ds>.json` — constant-mask floor, centroid stats, density map | C5 |
| A9 | External evaluation | guarded test loader, one token | best config per family | `reports/json/external/colondb.json` | C4 |
| A10 | Efficiency profile | `profiling/` | one checkpoint per distinct architecture/width | `reports/json/profiling.json` | C1 |

**A9 is a one-shot.** Run it last, after every other result is final. Spending the token and then
changing a config means declaring the repeat in the manuscript.

Not run for this paper (implemented but unused): Integrated Gradients full attribution, Seg-Grad-CAM,
ERF, CKA, uncertainty/retention, photometric corruptions. Keep them out of the repo branch's README
too.

---

## 10 · Statistical procedure

| Step | Method |
|---|---|
| Unit of analysis | Both reported: per-image (Wilcoxon) and per-seed (mean ± std, bootstrap CI over seed means). The paper states which is which |
| Paired test | Wilcoxon signed-rank on per-image Dice, within each declared family |
| Correction | Holm–Bonferroni within each family |
| Effect size | Cliff's delta + paired median difference, reported with every p-value |
| Null case | **TOST equivalence** against the 0.010 bound. Verdict ∈ {significant, equivalent_within_bound, inconclusive} |
| Cross-dataset ranking | Friedman + Nemenyi if ≥3 configs × 3 datasets are compared |
| Meaningfulness | `meaningfulness_gate` against MMD = 0.010; verdict string used verbatim in the paper |

**Comparison families** (declared in advance; nothing else carries an inferential claim):

- **F1** channel modes: m1 vs {m2, m3, m4, m5, m6, m7}
- **F2** capacity controls: each mode vs its width-matched RGB control
- **F3** order ablation: m4-post vs m4-pre; m5-post vs m5-pre
- **F4** shortcut: m8 vs constant-mask floor; translation degradation m1 vs m4
- **F5** generality: U-Net replication of F1/F2 headline rows

Anything outside F1–F5 is labelled **exploratory** in the manuscript text.

---

## 11 · Tables and figures

Six pages allows **3 tables and 3 figures**. Everything rendered through `reporting/` — nothing
assembled by hand. Blocking rules apply to all.

| ID | Content | Renderer | Source artefacts |
|---|---|---|---|
| **T1** | Channel modes: mode, effective channels, params, GFLOPs, Dice mean±std (seeds), 95% CI, vs-m1 corrected p, verdict | `render_channel_mode_table` | per-image Parquet + `stats/F1.json` + `profiling.json` |
| **T2** | Capacity controls: each mode paired with its width-matched control, delta, TOST verdict | `render_capacity_control_table` | + `stats/F2.json` |
| **T3** | Order ablation: m4/m5 post vs pre, per dataset | `render_order_ablation_table` | + `stats/F3.json` |
| **F1** | Shortcut. Panel A: coord-only Dice vs constant-mask floor per dataset, threshold line. Panel B: Dice vs translation magnitude, m1 vs m4 | `render_shortcut_figure` | A4, A6, A8 |
| **F2** | Attribution. Grouped bars: per-group Dice drop (occlusion) and Shapley mass, per dataset | `render_occlusion_figure` | A1, A2 |
| **F3** | Centre-bias scatter: index vs (m4 − m1) Dice gain, one point per dataset | `render_centre_bias_scatter` | A8 + F1 stats |

If space runs short, **F3 goes first** and becomes one sentence in the discussion.

Every table and figure carries a provenance footer (snapshot ID, git commit, generation date) in the
artefact, stripped for the camera-ready.

---

## 12 · Paper structure and page budget

| Section | Pages | Contents |
|---|---|---|
| I. Introduction | 0.75 | Gap: existing coordinate-channel studies are small, uncontrolled for capacity, and report inconsistent findings. Three contribution bullets. Explicit statement that no new architecture is proposed |
| II. Related work | 0.5 | CoordConv lineage, position encoding in CNNs, colour-space inputs, shortcut learning, evaluation methodology |
| III. Method | 1.25 | Channel construction with equations; the order distinction with a small diagram; capacity matching; three controls (width-matched, randproj, coord-only) |
| IV. Experimental setup | 0.4 | Datasets, splits, seeds, pre-registered thresholds, statistical procedure |
| V. Results | 2.0 | T1, T2, T3, F1, F2 (+F3 if space) |
| VI. Discussion & limitations | 0.75 | The practical guideline; why single-encoder; limitations |
| References | 0.35 | ~22 entries |

Results prose pattern: table → one paragraph on what it shows → one sentence on what it does not.
No narrative padding; there is no room.

**Firewall.** No sentence anywhere about multi-branch architectures, fusion, or where priors should
enter a network. Future work mentions more modalities and 3-D only.

---

## 13 · Results triage

Decide within one hour of seeing T1–T3, and do not relitigate.

| Outcome | Headline order | Framing |
|---|---|---|
| Order effect large; gains vanish under matching | **C3 → C1 → C4** | "A correctness pitfall and a capacity confound in coordinate-channel augmentation" — strongest, most positive |
| Order effect null; gains vanish under matching | **C1 → C4 → C2** | Controlled negative result with equivalence bounds. TOST is what makes this publishable |
| Gains survive matching | **C4 → C5 → C1** | "Input priors help — and here is how much of it is positional shortcut" |

All three are writable. That is the design property; do not treat any of them as failure.

---

## 14 · Submission checklist

- [ ] ≤ 6 pages, IEEE 2-column, figures and references included
- [ ] **No author names, affiliations, postal addresses, or email addresses** — violation causes immediate rejection
- [ ] PDF metadata scrubbed: `exiftool -all= paper.pdf` (the template inherits the OS username)
- [ ] Repo link points to `anonymous.4open.science`, never GitHub
- [ ] Leak grep clean on the anonymised repo: `mamba|ss2d|vss|cbffm|fusion|routing|auxiliary|dual.?encoder|dissert|thesis|<name>|<institution>`
- [ ] Figures legible at print size in a 2-column layout
- [ ] Every DOI in the reference list resolved through `https://doi.org/` (see §16)
- [ ] `ANALYSIS_PLAN.md` commit timestamp precedes the first training run
- [ ] Ledger `Test_Evals` shows exactly one row per (config, ColonDB)
- [ ] No reported result comes from a dirty-tree run

Post-acceptance: IEEE e-copyright, PDF eXpress-compatible camera-ready, at least one author registered.

---

## 15 · Schedule

| Day | Infrastructure | Runs | Analysis | Writing |
|---|---|---|---|---|
| 1 | T1 conference branch | T2, T3 (impl. plan) | T5 freeze plan, T4 | Outline, related work |
| 2 | Grep audit, verification | Pre-flight → launch Block A | T6, T7 | Method |
| 3 | T8 | Launch Blocks B, C | T9 renderers | Setup |
| 4 | — | Launch Block D | A1, A2, A3, A8 | Results scaffolding |
| 5 | — | Buffer / requeue | A4–A7, A10, **A9 last** | Full draft |
| 6 | Anonymised repo final check | — | Tables, figures, stats | Revision |
| 7 | — | — | — | Format check, metadata scrub, submit |

---

## 16 · References

**Verify every DOI through `https://doi.org/` before submission.** Entries marked ⚠ are ones I could
not confirm against a primary source in preparing this plan — resolve them first.

### Core prior art — coordinate and position priors

| # | Reference | Identifier | Status |
|---|---|---|---|
| 1 | Liu R. et al. *An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution.* NeurIPS 2018 | arXiv:1807.03247 | confirmed |
| 2 | El Jurdi R., Petitjean C., Honeine P., Abdallah F. *CoordConv-Unet: Investigating CoordConv for Organ Segmentation.* IRBM 42(6):415–423, 2021 | ⚠ DOI to verify | venue confirmed |
| 3 | El Jurdi R., Dargent T., Petitjean C., Honeine P., Abdallah F. *Investigating CoordConv for Fully and Weakly Supervised Medical Image Segmentation.* IPTA 2020, pp. 1–5 | ⚠ DOI to verify | venue confirmed |
| 4 | Islam M.A., Jia S., Bruce N.D.B. *How Much Position Information Do Convolutional Neural Networks Encode?* ICLR 2020 | arXiv:2001.08248 | ⚠ verify arXiv ID |

### Shortcut learning and validation methodology

| # | Reference | Identifier | Status |
|---|---|---|---|
| 5 | Lin M. et al. *Shortcut Learning in Medical Image Segmentation.* MICCAI 2024, LNCS 15008 | 10.1007/978-3-031-72111-3_59 · arXiv:2403.06748 | confirmed |
| 6 | Geirhos R. et al. *Shortcut Learning in Deep Neural Networks.* Nature Machine Intelligence 2:665–673, 2020 | ⚠ 10.1038/s42256-020-00257-z | verify |
| 7 | Isensee F. et al. *nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation.* MICCAI 2024 | arXiv:2404.09556 | confirmed |
| 8 | Maier-Hein L. et al. *Metrics Reloaded: Recommendations for Image Analysis Validation.* Nature Methods 21:195–212, 2024 | ⚠ 10.1038/s41592-023-02151-z | verify |
| 9 | Adebayo J. et al. *Sanity Checks for Saliency Maps.* NeurIPS 2018 | arXiv:1810.03292 | confirmed |

### Architectures

| # | Reference | Identifier | Status |
|---|---|---|---|
| 10 | Ronneberger O., Fischer P., Brox T. *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI 2015, LNCS 9351:234–241 | 10.1007/978-3-319-24574-4_28 | confirmed |
| 11 | Isensee F. et al. *nnU-Net: A Self-Configuring Method for Deep Learning-Based Biomedical Image Segmentation.* Nature Methods 18:203–211, 2021 | ⚠ 10.1038/s41592-020-01008-z | venue confirmed |
| 12 | Rahman M.M., Marculescu R. *MK-UNet: Multi-Kernel Lightweight CNN for Medical Image Segmentation.* ICCV 2025 Workshops (CVAMD) | arXiv:2509.18493 | verify published version |
| 13 | Oktay O. et al. *Attention U-Net: Learning Where to Look for the Pancreas.* MIDL 2018 | arXiv:1804.03999 | confirmed |
| 14 | Fan D.-P. et al. *PraNet: Parallel Reverse Attention Network for Polyp Segmentation.* MICCAI 2020 | ⚠ 10.1007/978-3-030-59725-2_26 | verify |

### Datasets

| # | Reference | Identifier | Status |
|---|---|---|---|
| 15 | Bernal J. et al. *WM-DOVA Maps for Accurate Polyp Highlighting in Colonoscopy (CVC-ClinicDB).* Computerized Medical Imaging and Graphics 43:99–111, 2015 | ⚠ 10.1016/j.compmedimag.2015.02.007 | verify |
| 16 | Tajbakhsh N., Gurudu S.R., Liang J. *Automated Polyp Detection in Colonoscopy Videos (CVC-ColonDB).* IEEE TMI 35(2):630–644, 2016 | ⚠ 10.1109/TMI.2015.2487997 | verify |
| 17 | Jha D. et al. *Kvasir-SEG: A Segmented Polyp Dataset.* MMM 2020 | ⚠ 10.1007/978-3-030-37734-2_37 | verify (cite only if used) |
| 18 | Codella N. et al. *Skin Lesion Analysis Toward Melanoma Detection 2018 (ISIC).* 2019 | arXiv:1902.03368 | confirmed |
| 19 | Tschandl P., Rosendahl C., Kittler H. *The HAM10000 Dataset.* Scientific Data 5:180161, 2018 | ⚠ 10.1038/sdata.2018.161 | verify (ISIC18 source data) |
| 20 | Al-Dhabyani W. et al. *Dataset of Breast Ultrasound Images (BUSI).* Data in Brief 28:104863, 2020 | ⚠ 10.1016/j.dib.2019.104863 | verify |

### Methods and tooling

| # | Reference | Identifier | Status |
|---|---|---|---|
| 21 | Lundberg S.M., Lee S.-I. *A Unified Approach to Interpreting Model Predictions (SHAP).* NeurIPS 2017 | arXiv:1705.07874 | confirmed |
| 22 | Buslaev A. et al. *Albumentations: Fast and Flexible Image Augmentations.* Information 11(2):125, 2020 | ⚠ 10.3390/info11020125 | verify |
| 23 | Demšar J. *Statistical Comparisons of Classifiers over Multiple Data Sets.* JMLR 7:1–30, 2006 | no DOI (JMLR open access) | confirmed |
| 24 | Lakens D. *Equivalence Tests: A Practical Primer for t-Tests, Correlations, and Meta-Analyses.* Social Psychological and Personality Science 8(4):355–362, 2017 | ⚠ 10.1177/1948550617697177 | verify |

**Reference budget:** ~22 entries fit in 0.35 pages. If over, cut in this order: 17 (if Kvasir unused),
19, 13, 14.

**Do not cite** in this paper: anything on Mamba, state-space models, dual-branch or multi-encoder
architectures, or feature fusion. Citing that literature signals the thesis direction.

---

## 17 · Risk register

| Risk | Severity | Trigger | Response |
|---|---|---|---|
| Double-blind breach via repo or PDF metadata | **Critical** | Any grep hit; any metadata field | Stop and fix. Automatic rejection otherwise |
| Six pages cannot hold the content | **Critical** | Draft exceeds 6 pages on day 6 | Cut F3, then C2 to two sentences, then compress related work |
| Width matching outside tolerance | High | Pre-flight | C1 dies. Fix the width-preset search before launching |
| Compute shortfall | High | Pre-flight extrapolation >80% budget | Cut Block D, then m3/m7 on two datasets. Never seeds |
| Order effect null | Medium | Block C results | Demote C3 to a paragraph; promote C4 to headline |
| Reviewer expects an accuracy improvement | Medium | — | Frame as a guideline in title and abstract, not a debunking |
| A DOI in the reference list does not resolve | Medium | Pre-submission check | Resolve or drop the citation. Never submit an unverified DOI |
| External token spent then config changed | Medium | — | Declare the repeat explicitly in the manuscript |

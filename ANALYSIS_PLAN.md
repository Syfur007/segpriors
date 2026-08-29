# Frozen Analysis Plan — committed 2026-08-30

## Primary endpoint
Mean per-image Dice, macro-averaged, on the held-out test split, for the
channel-mode × capacity-control comparison on MK-UNet, pooled across the three
training datasets.

## Comparison families (declared in advance)
F1 channel_modes      : m1 vs {m2, m3, m4, m5, m6, m7, m8}
F2 capacity_controls  : each mode vs its width-matched RGB control
F3 order_ablation     : m4-post vs m4-pre; m5-post vs m5-pre
F4 shortcut           : m8 (coord-only) vs constant-mask floor;
                        translation-shift degradation, m1 vs m4
F5 generality         : U-Net replication of F1 and F2 headline rows

## Thresholds (pre-registered, not to be changed after any run)
MMD (minimum meaningful Dice difference) : 0.010
Equivalence bound for TOST               : 0.010
Coordinate-only shortcut threshold       : 0.300 Dice
Attribution agreement threshold          : 0.700 (rank correlation)

## Tests
Per-image  : Wilcoxon signed-rank, Holm-Bonferroni within each family
Seed-level : mean +/- std, bootstrap CI over seeds (unit of analysis stated in paper)
Null case  : TOST equivalence against the bound above
Effect size: Cliff's delta, reported with every p-value

## Seeds
[1337, 2024, 7]  identical for every configuration

## Reporting commitment
All families above are reported regardless of outcome. Anything not listed here is
labelled exploratory in the manuscript and carries no inferential claim.

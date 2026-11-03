# Figures

Drop the paper figures here using exactly these filenames. The README and the
project page both reference them by these paths, so no other edits are needed
once the files exist.

Two figures previously in this directory (`overview.jpg`, `vis-examples.jpg`)
were REPA-E's and have been removed; they did not depict this method.

| Filename | Paper | Content | Used by |
| --- | --- | --- | --- |
| `GbR_github_vis.png` | new | Qualitative sample grid: text-to-image samples on top, class-conditional ImageNet below. **Present.** | README hero |
| `teaser.jpg` | Figure 1 | The four paradigms: (a) conventional two-stage, (b) naive end-to-end, (c) REPA-E, (d) ours. The headline figure. | project page hero |
| `genfirst.jpg` | new | The GenFirst schedule itself: phase 1 shapes the latent under a strong prior weight, phase 2 lowers it so reconstruction recovers detail. Referenced as the algorithm figure. | project page |
| `dynamics.jpg` | Figure 2 | Asymmetric learning dynamics: reconstruction converges in far fewer steps than generation. | project page |
| `latent_stats.jpg` | Figure 3 | Effect of each objective on the posterior, as Std(mu) vs E[sigma^2]. | project page |
| `scaling.jpg` | Figure 4 | Latent dimension, VAE capacity, and prior capacity, none of which resolve the conflict. | project page |
| `samples_256.jpg` | new | ImageNet 256x256 samples, SiT-XL/1 with guidance. | project page |
| `samples_512.jpg` | new | ImageNet 512x512 samples. | project page |
| `samples_t2i.jpg` | new | Text-to-image samples with prompts. | project page |

Recommendations: use JPEG for photographic sample grids and PNG for diagrams and
plots (rename the entries above to `.png` and update the two files if you do).
Keep each file under about 2 MB so the repository stays clonable, and target
around 1600 px wide, which is enough for a 2x display at typical page widths.

`samples_*.jpg` correspond to the qualitative figure the paper currently marks as
`Figurexxx`, so it does not exist yet.

Until a figure file exists, the project page shows a labelled dashed placeholder
box naming the missing file instead of a broken image, so the page is safe to
preview at any point.

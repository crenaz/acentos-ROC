# Accuracy

How this project measures OCR quality, and what the measurements say. Every default
in the pipeline is here because it was measured, and several plausible ideas are
recorded here as failures so they are not retried.

See also: [installation](installation.md) · [usage](usage.md) · [development](development.md)

> **Tesseract's confidence figures are its self-assessment, not accuracy.** A model
> can be confidently wrong. Treat confidence as a regression check — "did this change
> break something?" — and use character error rate against a manual transcription
> whenever a decision actually depends on which configuration is better.

## Ground truth (the metric that matters)

Measured as character error rate against a manual transcription of a Cayman job
listing, via `scripts/evaluate_cer.py`:

| Pipeline | `--psm 3` | `--psm 4` | `--psm 6` |
| --- | --- | --- | --- |
| Old binarising stack | 34.4% | 30.5% | 51.9% |
| Grayscale + blur3 | 8.9% | 8.9% | 16.2% |
| **Current: grayscale + deskew + blur3** | **7.3%** | 10.3% | 14.4% |

Three things worth knowing from those measurements:

- **Binarising was the single biggest quality problem.** Tesseract 5's LSTM engine
  binarises internally, and does it far better than a hand-tuned adaptive threshold.
  Removing `AdaptiveThresholdFilter` and `MorphologyFilter` from the default stack
  cut character error rate from 30.5% to 8.9%. Those filters remain available for
  explicit use, but the preprocessing they represent was written for the pre-LSTM era.
- `spa+eng` beats either language alone on *both* images, including the English
  one, which is why it is the default rather than plain `spa`.
- Cost of `tessdata_best` is roughly 2× wall clock — `document.png` takes ~3.7s
  versus ~1.8s with the system models on a 4-core i7-1165G7.

## Corpus evaluation

A single image cannot tell you whether a change helped — the first two
transcriptions pointed opposite ways on `--deskew`. `scripts/evaluate_corpus.py`
runs the whole transcribed corpus through a sweep of configurations:

```bash
export ACENTOS_CORPUS="/path/to/Cayman Job Clippings"
uv run python scripts/evaluate_corpus.py --psm 3 4 6 --deskew both --per-image
```

Pairing is by filename: an image `IMG_1594.JPEG` anywhere under the root is matched
with `text-of-IMG_1594.md`, also anywhere under the root. Images with no
transcription are listed rather than silently skipped, because a benchmark that
quietly shrinks stops meaning anything.

Not every photograph in a corpus directory is a sample, though — mastheads kept for
provenance, duplicates, reference shots. List those in a `.corpus-ignore` file at
the corpus root and they are excluded from both counts, so the warning keeps its
value without anyone having to delete a photo to silence it:

```
# Image stems deliberately excluded from the benchmark, one per line.
IMG_1600    # masthead: newspaper logos, records where the clippings came from
```

It reports two metrics side by side, and **the gap between them is the point**:

| Metric | Sensitive to order? | Answers |
| --- | --- | --- |
| CER | yes | what you actually get out of the pipeline |
| word miss rate | no | what the recogniser managed to read |

- Both low → the page was read correctly.
- Both high → a recognition failure; the words genuinely are not there.
- **CER high, word miss low** → the words were recognised and then emitted in the
  wrong order. That is a page-segmentation failure, and no amount of image
  preprocessing will fix it.

Rates are micro-averaged (errors and lengths summed separately) so a two-line
advert cannot outweigh a full page. `--json` writes the raw per-image rows for
comparison against a later run.

### Baseline

Across 15 transcribed Cayman job listings, `--lang eng`, 2026-08-08
(`results/baseline-2026-08-08.json`):

| Configuration | CER | word miss | confidence |
| --- | --- | --- | --- |
| **psm 3 + deskew + reading order** *(current default)* | **11.9%** | 10.4% | 88.1% |
| psm 3 + deskew | 18.9% | 10.7% | 88.1% |
| psm 6 | 22.3% | 12.1% | 84.4% |
| psm 6 + deskew | 22.3% | 11.9% | 84.3% |
| psm 4 + deskew | 23.6% | 11.2% | 87.4% |
| psm 3 *(default before 2026-08-08)* | 24.0% | 11.6% | 87.1% |
| psm 4 | 27.6% | 12.9% | 86.3% |

No single mode wins everywhere. Choosing the best configuration per image would
reach 15.8% — the ~3-point gap under the best fixed setting is what per-image
adaptation is worth, and it is much smaller than the 5-point deskew effect below.

Two images (`IMG_1599`, `IMG_1604`) sit near 30% in *every* configuration with
correspondingly high word miss rates. Those are recognition failures, and they are
**capture-limited rather than pipeline-limited** — both are dominated by perspective
and curvature rather than rotation, which a single global deskew cannot correct, and
`IMG_1604`'s text is only 31px tall.

Two remedies were tried and measured across the corpus. Neither works, so they are
recorded here rather than retried:

| Attempt | Corpus CER |
| --- | --- |
| baseline | **12.0%** |
| `--scale 1.5` | 12.7% |
| `--scale 2` | 12.6% |
| drop words below confidence 30 | 12.2% |
| drop words below confidence 60 | 13.3% |

Upscaling looks convincing on one sample — `IMG_1599` alone goes 32.2% → 22.9% at
×1.5 — and the corpus flatly contradicts it. The per-image pattern is scatter, not
signal: `IMG_1646` reaches 0.8% at ×2 while four other images get worse. That is
Tesseract's layout analysis reacting to a different input size, not a resolution
effect, and it is a good illustration of why single-image results are not evidence.

## Filter stacks compared

Stacks are composed on the command line (see
[Composing a pipeline](usage.md#composing-a-pipeline)) and scored against the whole
corpus. Measured with `--lang eng`, `--psm 3`, reading order on
(`results/stacks-2026-08-08.json`):

| Stack | CER | word miss | confidence |
| --- | --- | --- | --- |
| **`grayscale deskew blur:ksize=3`** *(default)* | **12.0%** | 10.7% | 88.1% |
| `grayscale deskew blur:ksize=5` | 12.4% | 11.6% | 88.0% |
| `grayscale deskew` *(no blur)* | 12.4% | 11.0% | 87.4% |
| `grayscale deskew clahe blur:ksize=3` | 18.1% | 12.1% | 85.4% |
| `grayscale deskew blur:ksize=5 threshold morphology` *(the original stack)* | 27.1% | 21.3% | 69.6% |

Three things fall out of that:

- **The blur is nearly irrelevant.** Dropping it entirely costs 0.4 points, and
  widening it to 5 costs the same. It is the least consequential knob in the stack.
- **CLAHE hurts, badly and unevenly.** +6.1 points overall, and it destroys
  `IMG_1648` specifically (4.3% → 48.5%). Boosting local contrast on newsprint
  amplifies paper texture along with the ink.
- **The original binarising stack is confirmed catastrophic at corpus scale.**
  27.1% against 12.0%, and it fails one image completely (`IMG_1598`, 100% CER).
  That was previously measured on a single image; the corpus agrees.

## Deskew (`--deskew`)

**On by default** as of 2026-08-08 — worth 5.1 points corpus-wide (24.0% → 18.9% CER
at psm 3) for negligible extra time. Pass `--no-deskew` to disable. Per-image delta:

| Image | CER change with `--deskew` |
| --- | --- |
| `IMG_1648` | **−40.9%** |
| `IMG_1658` | −18.8% |
| `IMG_1601` | −9.5% |
| `IMG_1603` | −6.8% |
| `IMG_1594` | −1.6% |
| `IMG_1657` | +4.3% |
| `IMG_1595` | +7.5% |
| 8 others | unchanged — no confident angle found |

Four large wins against two modest regressions, because on more than half the corpus
the filter finds no confident angle and declines to act rather than guessing.

Note what the largest win actually is. `IMG_1648` improves by 41 points not because
its characters were unreadable, but because its skew made Tesseract's layout analysis
invent column boundaries and shred the page into out-of-order blocks — 45.7% CER
against a 5.4% word miss rate. Straightening the page fixed the segmentation. Skew
damage and reading-order damage are frequently the same bug.

## Reading order (`--reading-order`)

**On by default.** Corpus CER 18.9% → 12.0%, at no extra OCR cost.

Tesseract's page segmentation mishandles the layout these adverts almost always
use — a full-width headline and intro, a two-column body, then a full-width
footer — and it fails in *both* directions depending on skew:

| | Block widths | Failure |
| --- | --- | --- |
| skewed page | narrow | applies the column split to the **whole page**, shredding full-width paragraphs into three pieces emitted far apart |
| straight page | full | reads the **two-column body as full width**, interleaving left and right line by line |

Either way the characters are read correctly and put in the wrong order, which
CER punishes exactly as hard as not reading them at all. On `IMG_1595`: 40.5% CER
against a 7.9% word miss rate.

The fix re-segments the page geometrically and re-emits the words. Every word
already comes back from the OCR pass with a bounding box, so this needs no second
pass. The segmentation is an XY-cut over a map of those boxes — not over pixels,
which sidesteps the page border, uneven lighting and newsprint texture that make
pixel-based layout analysis fragile on a photograph.

Three details carry the benefit:

- **Cut one gap at a time, widest first.** Splitting at every qualifying gap at
  once breaks the page into horizontal bands *before* anything notices the gutter,
  and then splits each band into left and right independently — which interleaves
  the columns a few lines at a time, reproducing the bug being fixed.
- **Assign words to regions, but keep Tesseract's lines inside them.** The two
  failure modes are mirror images and both need handling: a line Tesseract split
  across blocks must be rejoined, and a line it merged across the gutter must be
  torn apart. Per-word assignment does the tearing; grouping by Tesseract's own
  line number inside each region does the rejoining.
- **Never re-derive lines from word positions.** Tesseract tracks a baseline per
  line, so it stays correct on a photographed page whose lines sag across the
  frame. Clustering words by vertical centre does not — it drops the last words of
  a sloping line onto the line below. An earlier version did exactly that and cost
  `IMG_1594` 7.3% → 18.8%.

Corpus result: nine images improve, most dramatically `IMG_1595` 48.0% → **4.7%**,
`IMG_1646` 23.2% → **7.3%** and `IMG_1596` 17.9% → **6.0%**. Four are untouched.
One regresses: `IMG_1658` 4.7% → 9.3%, where a short trailing line becomes its own
region and moves ahead of the line above it.

## Reference baselines

With the default filter stack at `--psm 6`, the committed sample images score:

| Configuration | `fluoxetine.png` | `document.png` |
| --- | --- | --- |
| Tesseract 4.1.1, system models, `--lang eng`, binarising stack | 75.30% | 49.37% |
| Tesseract 4.1.1, system models, `--lang spa+eng`, binarising stack | 81.73% | 56.21% |
| Tesseract 5.5.1, system models, `--lang spa+eng`, binarising stack | 84.44% | 61.23% |
| Tesseract 5.5.1, `tessdata_best`, `--lang spa+eng`, binarising stack | 82.74% | 64.47% |
| **Current default** — grayscale + deskew + blur3, `--psm 3` | **83.69%** | **95.17%** |

Both samples are flat-scanned, so the deskew estimator finds no confident angle and
declines to act; enabling it by default left these two figures untouched.

These are confidence figures, and per the warning at the top of this page they are a
regression check rather than a measure of accuracy. Re-run them after any change to
`uv.lock`, since OCR output is sensitive to the OpenCV version.

---

Recorded against Tesseract 5.5.1 and the current `uv.lock`.

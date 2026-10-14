# Evaluation protocol for AI Evidence Extractor

## Purpose

Measure how well the video detector finds and localizes objects in a fixed,
manually annotated set. Treat outputs as review leads; detector scores and
threat labels are not findings of fact or assessments of a person's intent.

## Annotation format

Create UTF-8 CSV files with these columns:

```csv
video,timestamp,label,x1,y1,x2,y2
camera_a.mp4,12.40,firearm,410,128,538,265
```

Use source-video timestamps in seconds and pixel coordinates in the original
frame. Annotate every visible target in sampled frames, including frames with
no target (those frames are still needed to assess false alarms; record them in
a separate frame manifest). Agree on a label guide before annotation. Have a
second reviewer independently label a subset and resolve disagreements before
scoring.

## Sampling and split

- Sample across videos, scenes, lighting, camera quality, distance, occlusion,
  motion blur, and benign lookalikes. Include negative footage with no target.
- Keep all frames from a video in one split. Do not tune thresholds on the
  held-out test set.
- Publish counts by video, class, positive/negative frame, and condition.
- Record model repository/revision, confidence thresholds, fusion settings,
  sampling FPS, resolution, hardware, and software versions.

## Metrics and comparison

Run `python evaluation.py ground_truth.csv predictions.csv`. A prediction is
matched one-to-one to a same-class annotation when timestamp distance is at
most 0.5 seconds and box IoU is at least 0.5. Report TP, FP, FN, precision,
recall, and F1 overall and per class. Also report false alarms per minute on
negative videos, missed targets, processing time, and results at stricter and
looser thresholds. Compare the ensemble with each constituent detector using
the same split and sampling. Do not claim the ensemble is better without these
results.

## Reviewer workflow and traceability

For every candidate, retain the original video name, source timestamp, bounding
box, predicted class, confidence, contributing model(s), and processing
configuration. The reviewer should be able to open the corresponding original
clip and mark each item **confirmed**, **uncertain**, or **false positive**;
keep reviewer decisions separate from model predictions in exports. Preserve
the original media and record its SHA-256 hash when chain-of-custody matters.

## Explainability and limitations

EigenCAM is an exploratory visualization of model activation, not proof that a
box or class is correct. Threat scores are heuristic prioritization values,
not calibrated probabilities. State these limits in the interface and report.
Document failures, unsupported conditions, and the fact that automated output
requires human verification.

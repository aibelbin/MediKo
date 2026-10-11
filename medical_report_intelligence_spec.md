# Medical Report Intelligence System --- Project Specification

**Status:** Initial specification\
**Project type:** Biomedical NLP research prototype\
**Primary goal:** Extract structured clinical information from medical
reports and rigorously evaluate how well a biomedical NER model handles
noisy text.

## 1. Project Purpose

Build a research prototype that accepts medical reports as PDF or plain
text, extracts clinically relevant information, and presents the results
in a readable dashboard.

The project must demonstrate a genuine, measurable ML component. Model
predictions, confidence scores, evaluation metrics, and error-analysis
examples must come from actual runs---not fabricated or manually
invented results.

**This is an information-extraction research tool, not a diagnostic
system. It must not provide diagnoses, treatment recommendations, or be
used for medical decision-making.**

## 2. Long-Term Goals

1.  Extract text from digital PDFs and scanned PDFs.
2.  Normalize common text noise without silently changing clinical
    meaning.
3.  Fine-tune a biomedical Transformer for medical named entity
    recognition (NER).
4.  Identify relevant entity types, including diseases, symptoms,
    medications, investigations, and medical findings.
5.  determine whether each entity is affirmed, negated, or uncertain in
    context.
6.  Extract relationships between entities where supported by the text.
7.  Extract dates and organize documented events into a chronological
    timeline.
8.  Show entities, context, status, confidence, relations, and timeline
    in a Streamlit dashboard.
9.  Compare a baseline NER system with the proposed noise-robust
    approach using reproducible experiments.
10. Document errors, limitations, and results honestly.

## 3. Intended Users and Use Case

A researcher or authorized user uploads a medical report and receives a
structured summary of information explicitly documented in that report.
The output helps organize information; it does not interpret the report
as a clinician would.

## 4. Proposed Processing Pipeline

Implement this as a normal, sequential Python pipeline. **Do not
introduce an LLM agent or orchestration framework unless a concrete
requirement justifies it.**

1.  **Input** --- accept PDF or TXT; validate file type and size.
2.  **Text extraction** --- extract embedded text from digital PDFs.
3.  **OCR fallback** --- use OCR for scanned pages or pages with
    insufficient extractable text.
4.  **Text normalization** --- normalize whitespace, line breaks, common
    OCR artifacts, and safe spelling/abbreviation variants. Preserve the
    original text and record transformations.
5.  **Sentence/section processing** --- split text while retaining
    offsets and useful context.
6.  **Medical NER** --- run the fine-tuned biomedical Transformer to
    identify spans and entity types.
7.  **Entity post-processing** --- merge subword/token predictions into
    character spans; validate and deduplicate spans without losing
    source context.
8.  **Negation detection** --- classify entity mentions as `PRESENT`,
    `NEGATED`, or `UNCERTAIN`, using a suitable clinical NLP tool or a
    documented baseline method.
9.  **Relation extraction** --- identify supported relations between
    extracted entities. Begin with a small, explicitly defined relation
    schema.
10. **Temporal extraction** --- detect dates or time expressions and
    associate events with them when supported by the report.
11. **Timeline generation** --- sort dated events chronologically; label
    missing or ambiguous dates instead of guessing.
12. **Structured output** --- serialize results to a documented JSON
    schema.
13. **Dashboard** --- display extracted text, highlighted spans, entity
    details, statuses, confidence values, relations, timeline, and
    evaluation results.
14. **Evaluation and error analysis** --- run the defined experiments
    and save machine-readable predictions and metrics.

Stages may use deterministic code, established NLP libraries, or
pretrained tools. The only component that must be fine-tuned for the
initial research contribution is the medical NER model.

## 5. NER Model

Start by comparing a pretrained biomedical Transformer such as
**BioBERT** or **PubMedBERT**. Choose one as the initial model after
checking dataset compatibility, tokenizer behavior, licensing, and
available compute.

Fine-tune the model for token classification using BIO tags, for
example:

-   `B-DISEASE`, `I-DISEASE`
-   `B-SYMPTOM`, `I-SYMPTOM`
-   `B-MEDICATION`, `I-MEDICATION`
-   `B-INVESTIGATION`, `I-INVESTIGATION`
-   `B-FINDING`, `I-FINDING`
-   `O` for tokens outside target entities

Example sentence: `Patient has persistent fever.`

-   `Patient` → `O`
-   `has` → `O`
-   `persistent` → `B-SYMPTOM`
-   `fever` → `I-SYMPTOM`

The exact label set must be finalized before annotation/training. Avoid
adding labels that the chosen dataset cannot support reliably.

## 6. Dataset and Annotation Requirements

-   Select a suitable biomedical/clinical NER dataset after reviewing
    its label schema, data format, license, and usage restrictions.
-   Do not assume that a general biomedical literature dataset
    automatically represents real-world hospital reports.
-   Record the dataset source, version, license, label mapping, and
    preprocessing decisions.
-   Split data into train, validation, and test sets before training.
    Prevent document or patient leakage where applicable.
-   Keep the test set separate from model selection and tuning.
-   If creating synthetic noise, retain the original clean examples and
    document every noise transformation.
-   Never present synthetic or generated examples as real patient
    records.
-   Do not upload private or identifiable patient data to external
    services without explicit authorization and appropriate safeguards.

## 7. Noise-Robustness Research

The main research question is:

> Does the proposed normalization/noise-robust approach preserve medical
> entity extraction performance better than a baseline when clinical
> text is noisy?

Compare at least these conditions:

1.  **Clean text** --- original held-out test examples.
2.  **Noisy text** --- the same examples with documented noise applied.
3.  **Noisy text + normalization** --- noisy examples after the
    normalization module.

Use identical underlying examples across conditions so comparisons are
meaningful. Include realistic, controlled noise types such as spelling
errors, OCR character substitutions, broken line breaks, whitespace
problems, and selected abbreviation variations. Do not alter clinical
meaning intentionally.

Define the baseline clearly. A useful starting point is the same
pretrained model fine-tuned without the proposed noise
augmentation/normalization strategy. Keep model architecture and data
splits comparable where possible. State exactly what differs between
systems.

## 8. Evaluation

For NER, calculate from actual held-out predictions:

-   Entity-level precision, recall, and F1
-   Per-entity-class precision, recall, and F1
-   Exact-span matching as the primary scoring rule
-   Confusion/error breakdown where applicable

For additional pipeline components, evaluate only when suitable labeled
examples or a defensible reference set are available:

-   Negation/uncertainty classification: class-wise metrics and
    confusion matrix
-   Relation extraction: precision, recall, and F1 against annotated
    relations
-   Temporal extraction/timeline: date/event extraction accuracy and
    association accuracy

Include: - Dataset sizes and split details - Model and tokenizer
identifiers/versions - Training configuration and random seed -
Evaluation scripts and raw prediction files -
Clean/noisy/normalized-noisy comparisons - Representative true
positives, false positives, and false negatives - Error analysis grouped
by noise type and entity class

Do not promise that the proposed model will outperform the baseline.
Report the observed results, including negative or inconclusive results.

## 9. Confidence Scores

Show model-derived confidence values and explain what they mean. Do not
treat token-level softmax probabilities as automatically calibrated
probabilities of clinical correctness. If confidence calibration is not
implemented and evaluated, label values as model confidence scores
rather than calibrated certainty.

## 10. Initial Output Schema

Use a versioned JSON schema. A possible starting structure:

``` json
{
  "document_id": "generated-id",
  "entities": [
    {
      "text": "fever",
      "entity_type": "SYMPTOM",
      "start_char": 24,
      "end_char": 29,
      "status": "PRESENT",
      "confidence": 0.91,
      "context": "Patient has persistent fever."
    }
  ],
  "relations": [],
  "timeline": [],
  "processing_metadata": {
    "ocr_used": false,
    "normalization_applied": true,
    "model_version": "recorded-at-runtime"
  }
}
```

The values above are illustrative schema examples only, not real model
results. Use actual offsets and outputs at runtime. Define how
overlapping entities, unknown dates, and missing confidence values are
represented.

## 11. Streamlit Dashboard

The dashboard should provide:

-   PDF/TXT upload and extracted-text preview
-   Highlighted entities in their original context
-   Entity type, status, and model confidence
-   Entity list with filtering by type/status
-   Extracted relations, when available
-   Chronological timeline with uncertain or missing dates clearly
    marked
-   Research/evaluation view for metrics and error analysis
-   Clear warning that the tool is not for diagnosis or medical
    decisions

The dashboard must not invent results when a component fails or has not
been implemented. Display an explicit unavailable/error state instead.

## 12. Suggested Implementation Phases

### Phase 1 --- Repository and data

-   Create project structure and environment setup.
-   Select dataset and document its license and label mapping.
-   Define the annotation schema and evaluation protocol.

### Phase 2 --- Baseline NER

-   Implement tokenization, BIO label alignment, fine-tuning, inference,
    and entity-span reconstruction.
-   Save checkpoints and evaluation predictions.
-   Establish baseline metrics.

### Phase 3 --- Noise experiments

-   Implement controlled noise generation and conservative
    normalization.
-   Evaluate clean, noisy, and noisy-plus-normalization conditions.
-   Run error analysis and save results.

### Phase 4 --- Supporting extraction modules

-   Add negation/uncertainty detection.
-   Add a defined relation schema and relation extraction baseline.
-   Add temporal extraction and timeline generation.
-   Keep unsupported or unreliable results explicitly marked.

### Phase 5 --- Input processing and interface

-   Add digital PDF extraction and OCR fallback.
-   Integrate the sequential pipeline.
-   Build the Streamlit dashboard.

### Phase 6 --- Reproducibility and documentation

-   Add tests, configuration, logs, model/data provenance, and
    reproducible commands.
-   Document limitations, privacy precautions, and evaluation results.

## 13. Engineering Requirements

-   Python-first implementation with modular, testable components.
-   Keep data processing, model training, inference, evaluation, and UI
    code separate.
-   Use configuration files for model names, paths, labels, thresholds,
    and experiment settings.
-   Make training and evaluation reproducible where practical.
-   Store model artifacts and experiment outputs separately from source
    code.
-   Add unit tests for text offsets, BIO alignment, span reconstruction,
    normalization, and JSON validation.
-   Preserve source text so users can verify each extracted item.
-   Do not silently discard uncertain, unsupported, or failed
    extractions.
-   Do not add multiple large models or an LLM orchestration layer
    without a demonstrated need.

## 14. Privacy, Safety, and Limitations

-   This project is for information extraction and research only.
-   It must not diagnose conditions, recommend treatment, or make
    clinical decisions.
-   Medical information can be sensitive. Prefer de-identified data and
    local processing where practical.
-   Do not send medical documents to external APIs unless the data use
    is authorized and the privacy implications are understood.
-   OCR, abbreviations, negation scope, relations, and date associations
    can fail. Surface uncertainty and retain source context.
-   Extracted statements must remain traceable to the source report; do
    not infer undocumented facts.

## 15. Definition of Done

The initial research prototype is complete when:

1.  A user can submit a supported PDF or TXT file.
2.  Text extraction/OCR and the sequential processing pipeline work.
3.  A fine-tuned NER model produces source-aligned entity spans.
4.  Baseline and proposed approaches are evaluated on the same held-out
    examples under the defined noise conditions.
5.  Metrics are generated by scripts from actual predictions.
6.  Error-analysis examples can be traced to real model outputs.
7.  Supporting modules return structured outputs and explicitly handle
    failures/uncertainty.
8.  The dashboard presents results without implying diagnosis.
9.  Setup, training, inference, and evaluation instructions are
    documented.

## 16. Instructions for the Coding Assistant

Before implementing features:

1.  Read this specification and summarize the proposed architecture.
2.  Inspect the existing repository before creating or overwriting
    files.
3.  Propose a minimal repository structure and implementation plan.
4.  Identify unresolved decisions---especially dataset choice, label
    schema, and baseline definition---rather than guessing.
5.  Implement incrementally, starting with dataset/schema and baseline
    NER.
6.  Add tests and runnable commands for each phase.
7.  Never fabricate metrics, model outputs, datasets, or completed
    features.
8.  Keep the project focused: a reproducible NER/noise-robustness
    experiment first, with the rest of the pipeline added incrementally.

# PerturbSelect

**Benchmarking functional selectivity in virtual-cell perturbation models**

## Motivation

Virtual-cell and perturbation-prediction models are commonly evaluated by how accurately they reconstruct molecular responses.

For therapeutic discovery, however, an intervention should not merely produce a strong effect. Its effect should be context-selective: for example, strongly impairing a malignant cell while relatively sparing a non-malignant cell.

PerturbSelect asks whether predicted perturbation responses preserve the molecular information required to identify such functionally selective interventions.

## Central question

Can perturbation-model predictions recover differential functional effects across cellular contexts?

## Phase 0: Biological premise

Before evaluating any virtual-cell model, we test whether experimentally measured perturbation transcriptomes themselves contain enough information to predict differential functional fitness.

Initial system:

- K562 leukemia cells
- RPE1 non-transformed cells
- Replogle et al. CRISPRi Perturb-seq
- matched pooled growth-screen phenotypes

If measured transcriptional responses cannot robustly predict differential fitness, the project will not proceed to virtual-cell benchmarking.

## Status

Phase 0 — biological-premise pilot.
---
layout: default
title: KB Manager Technical Documentation
description: Persian RAG Knowledge Base — Technical Reports
---

# KB Manager Technical Documentation

Welcome to the technical documentation for **KB Manager** — a Persian (Farsi) credit scoring knowledge base with dense semantic retrieval.

## Quick Links

- [📊 Technical Report](technical-report.html) — Embedding model, retrieval methods, evaluation metrics with LaTeX formulas
- [🚀 Quick Start](../README.md#quick-start) — Installation and usage
- [🏗 Architecture](../README.md#architecture) — System overview
- [📖 API Reference](../README.md#web-ui) — Web UI endpoints

## Overview

KB Manager is a complete pipeline for ingesting, preprocessing, chunking, embedding, and managing a knowledge base for RAG agents. It features:

- **Hybrid Retrieval**: BM25 (lexical) + Dense semantic (MiniLM multilingual) with RRF fusion
- **Persian-Native**: ZWNJ normalization, Hazm tokenization, typo handling
- **Versioned**: Document snapshots with rollback support
- **Observable**: Interactive Chart.js dashboards, benchmark automation

## Latest Benchmark (v3)

| Metric | Value | Δ vs v2 |
|--------|-------|---------|
| **Hit@5** | 89.2% | −0.8% |
| **Top-1** | **72.5%** | **+9.2%** |
| **MRR** | **0.787** | **+6.9%** |
| **Avg Latency** | **1.9 s** | **−34%** |

[View detailed technical report →](technical-report.html)

---

*Private — ICS Credit Scoring*
# Resume2Vec — ATS Resume Optimizer

Resume2Vec is a Generative AI project for optimizing resumes for Applicant Tracking Systems (ATS).

The system:
- parses resumes from PDF, DOCX, or text
- compares them against a job description
- computes an ATS score
- identifies missing mandatory and optional requirements
- uses a fine-tuned semantic model for deeper matching
- generates an ATS-oriented optimized draft
- shows before/after comparison and text diff

---

## Overview

Traditional ATS systems often rely heavily on exact keyword matching. Resume2Vec improves on this by combining:

- structured resume parsing
- fine-tuned semantic similarity
- mandatory vs optional requirement weighting
- ATS scoring
- safe optimization and explanation

This allows the system to handle both:
- exact matches (`ASP.NET` ↔ `ASP.NET`)
- related matches (`MAUI` ↔ mobile / Android-related work)

---

## Architecture

```text
Resume (PDF / DOCX / text) + Job Description
                    │
                    ▼
              [ Resume Parser ]
                    │
                    ▼
          [ Section Extraction Layer ]
     - skills
     - education
     - experience
     - formatting indicators
                    │
                    ▼
      [ Fine-Tuned Semantic Matcher ]
   - trained on resume dataset
   - category-aware / self-supervised similarity
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
 [ ATS Keyword / Skill Match ]  [ Semantic Scoring ]
        │                       │
        └───────────┬───────────┘
                    ▼
             [ ATS Scoring Engine ]
     - keyword score
     - skill score
     - semantic score
     - section quality score
     - format score
     - achievement score
                    │
                    ▼
          [ Resume Optimization Layer ]
     - professional summary generation
     - role alignment block
     - missing requirement suggestions
     - ATS-oriented draft generation
                    │
                    ▼
           [ Before / After Comparison ]
     - original score
     - optimized score
     - suggestions
     - text diff
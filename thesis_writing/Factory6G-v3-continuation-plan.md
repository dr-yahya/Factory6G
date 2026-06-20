# Factory6G v3 Continuation Plan

Source constraint: this plan starts from `thesis_writing/Factory6G-v3-Current-Draft.docx` only. It does not use later thesis attempts as source material.

## Overall Balance

The completed draft should remain near a 120-page ceiling by keeping the final body to roughly 30,000-34,000 words with visual support. Existing v3 Chapters 1-2 account for about 16,700 extracted words including the premature conclusion and references. The continuation therefore adds a compact but complete Chapter 3 completion, a results chapter organized by evidence family, a discussion chapter, and a concise conclusion.

| Chapter | Target pages | Purpose | Evidence and visual rhythm |
|---|---:|---|---|
| 1 Introduction | 10-12 | Establish 6G smart-factory motivation, questions, contribution scope | Keep existing v3 content; update organization only |
| 2 Literature Review | 24-28 | Position PHY, MAC, ML, and cross-layer gaps | Keep existing v3 content; do not expand unless citation cleanup is required |
| 3 Methodology | 26-30 | Connect the simulator, mathematical model, datasets, estimators, and schedulers | Architecture diagrams, equations, one algorithm, method matrix |
| 4 Results | 24-28 | Present completed simulations by family without mixing unrelated evidence | BER/latency/throughput plots, compact comparison tables |
| 5 Discussion | 14-18 | Interpret trends, trade-offs, limitations, and research implications | Synthesis table and design-rule discussion |
| 6 Conclusion | 6-8 | Answer the research questions and define future work | Contribution summary and future-work roadmap |
| References | 8-12 | Preserve verified v3 bibliography | Existing Zotero-style references retained |

## Chapter 3 Structure Plan

### 3.4 Channel and Link-Level Modelling
Purpose: provide the mathematical bridge from the existing system architecture to measurable BER. Contribution: formalizes the baseband signal model, channel families, noise control, and Eb/N0 sweep. Flow: start with the received-signal equation, then describe Rayleigh/Rician/TR 38.901 UMi assumptions, then define BER and confidence-aware interpretation. Evidence: code-backed simulator architecture and result schema. Connection: prepares the reader for estimator and resource-manager experiments.

### 3.5 Channel Estimator Methodology
Purpose: explain how LS, DFT, adaptive, PSO, and neural estimators are compared fairly. Contribution: frames estimator choice as an industrial reliability decision rather than only a signal-processing subproblem. Flow: describe shared Monte Carlo context, estimator families, expected error sources, and deployment cost. Evidence: estimator result CSVs and BER/latency/runtime plots. Connection: directly leads to Chapter 4 estimator results.

### 3.6 Resource-Manager and Cross-Layer Methodology
Purpose: define scheduling/power allocation as the MAC-layer control surface. Contribution: shows how resource directives connect PHY reliability feedback to MAC decisions. Flow: define state, action, reward/utility, baselines, and learned policies. Evidence: resource-manager comparisons across Rayleigh, Rician, and TR 38.901 UMi. Connection: supports resource-manager results and discussion.

### 3.7 BER-Oriented DRL Formulation
Purpose: motivate BER-first learning as a reliability-oriented policy. Contribution: gives the thesis an original AI/ML mechanism tied to smart-factory reliability. Flow: MDP definition, reward design, inference loop, and claim boundary. Evidence: BER-DRL checkpoint outputs and cross-channel comparison tables. Connection: prepares the BER-DRL results without overclaiming dominance.

### 3.8 Reproducibility, Traceability, and Page-Controlled Evidence Design
Purpose: document how results are generated, named, interpreted, and limited. Contribution: strengthens thesis credibility by separating simulated measurements from synthetic/anchored summaries. Flow: list outputs, metrics, and confidence rules. Evidence: `summary_v2.csv`, `stage_results_v2.csv`, and generated plots. Connection: provides the reading protocol for Chapter 4.

## Chapter 4 Structure Plan

Purpose: report results in evidence families: estimator reliability, channel sensitivity, modulation sensitivity, resource management, BER-DRL, and JIDD-SCMA. Expected contribution: demonstrate that reliability in smart factories depends on joint PHY/MAC decisions and channel realism, not a single isolated model. Logical flow: simple BER trends first, then harsher channels, then estimator/resource-manager trade-offs, then advanced learned/joint-processing evidence. Required evidence: plots from `reports/plots`, `system_design`, and current `results` folders; compact statistical tables from current CSVs. Connection: provides the empirical basis for Chapter 5.

## Chapter 5 Structure Plan

Purpose: synthesize rather than repeat results. Expected contribution: translate observed performance into design principles for 6G smart-factory communication. Logical flow: reliability mechanisms, multipath limitations, ML benefits, scheduler trade-offs, deployment implications, limitations. Required evidence: Chapter 4 trends plus literature links from v3 references. Connection: prepares the final research answers in Chapter 6.

## Chapter 6 Structure Plan

Purpose: close the thesis by answering the research questions and presenting bounded contributions. Expected contribution: state what was demonstrated, what remains simulation-bound, and what future experiments should prioritize. Logical flow: findings, contributions, practical implications, limitations, future work. Required evidence: concise summary of the implemented simulator and completed result families. Connection: final thesis closure before references.

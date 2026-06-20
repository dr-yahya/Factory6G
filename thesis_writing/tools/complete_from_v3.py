#!/usr/bin/env python3
from __future__ import annotations

import copy
import csv
import re
import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image


ROOT = Path("/app")
SRC_DOCX = ROOT / "thesis_writing" / "Factory6G-v3-Current-Draft.docx"
OUT_DOCX = ROOT / "thesis_writing" / "Factory6G-v3-Completed-From-Current-Progress.docx"
PLAN_MD = ROOT / "thesis_writing" / "Factory6G-v3-continuation-plan.md"
TMP_DIR = ROOT / "thesis_writing" / ".build_from_v3_docx"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
PIC = "http://schemas.openxmlformats.org/drawingml/2006/picture"
REL = "http://schemas.openxmlformats.org/package/2006/relationships"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"

NS = {"w": W, "r": R}
ET.register_namespace("w", W)
ET.register_namespace("r", R)
ET.register_namespace("wp", WP)
ET.register_namespace("a", A)
ET.register_namespace("pic", PIC)


def qn(ns: str, tag: str) -> str:
    return f"{{{ns}}}{tag}"


def w_el(tag: str, attrs: dict[str, str] | None = None, text: str | None = None) -> ET.Element:
    elem = ET.Element(qn(W, tag), attrs or {})
    if text is not None:
        elem.text = text
    return elem


def sub(parent: ET.Element, tag: str, attrs: dict[str, str] | None = None, text: str | None = None) -> ET.Element:
    elem = w_el(tag, attrs, text)
    parent.append(elem)
    return elem


def paragraph_text(p: ET.Element) -> str:
    return "".join(t.text or "" for t in p.iter(qn(W, "t"))).strip()


def p_style(p: ET.Element) -> str:
    style = p.find("./w:pPr/w:pStyle", NS)
    return style.get(qn(W, "val"), "") if style is not None else ""


def set_p_text(p: ET.Element, text: str, *, bold: bool = False) -> None:
    for child in list(p):
        p.remove(child)
    run = sub(p, "r")
    if bold:
        rpr = sub(run, "rPr")
        sub(rpr, "b")
    t = sub(run, "t", {qn("http://www.w3.org/XML/1998/namespace", "space"): "preserve"})
    t.text = text


def make_p(text: str = "", style: str | None = None, *, bold: bool = False, italic: bool = False) -> ET.Element:
    p = w_el("p")
    if style:
        ppr = sub(p, "pPr")
        sub(ppr, "pStyle", {qn(W, "val"): style})
        if style.startswith("Heading"):
            sub(ppr, "spacing", {qn(W, "before"): "180", qn(W, "after"): "100"})
        elif style == "Caption":
            sub(ppr, "jc", {qn(W, "val"): "center"})
            sub(ppr, "spacing", {qn(W, "after"): "120"})
    elif text:
        ppr = sub(p, "pPr")
        sub(ppr, "spacing", {qn(W, "after"): "120", qn(W, "line"): "276", qn(W, "lineRule"): "auto"})
        sub(ppr, "jc", {qn(W, "val"): "both"})
    if text:
        r = sub(p, "r")
        if bold or italic:
            rpr = sub(r, "rPr")
            if bold:
                sub(rpr, "b")
            if italic:
                sub(rpr, "i")
        t = sub(r, "t", {qn("http://www.w3.org/XML/1998/namespace", "space"): "preserve"})
        t.text = text
    return p


def make_table(rows: list[list[str]], widths: list[int]) -> ET.Element:
    tbl = w_el("tbl")
    tbl_pr = sub(tbl, "tblPr")
    sub(tbl_pr, "tblStyle", {qn(W, "val"): "TableGrid"})
    sub(tbl_pr, "tblW", {qn(W, "w"): str(sum(widths)), qn(W, "type"): "dxa"})
    borders = sub(tbl_pr, "tblBorders")
    for name in ["top", "left", "bottom", "right", "insideH", "insideV"]:
        sub(borders, name, {qn(W, "val"): "single", qn(W, "sz"): "4", qn(W, "space"): "0", qn(W, "color"): "BFBFBF"})
    grid = sub(tbl, "tblGrid")
    for width in widths:
        sub(grid, "gridCol", {qn(W, "w"): str(width)})
    for ridx, row in enumerate(rows):
        tr = sub(tbl, "tr")
        for cidx, cell_text in enumerate(row):
            tc = sub(tr, "tc")
            tc_pr = sub(tc, "tcPr")
            sub(tc_pr, "tcW", {qn(W, "w"): str(widths[cidx]), qn(W, "type"): "dxa"})
            if ridx == 0:
                shd = sub(tc_pr, "shd", {qn(W, "fill"): "D9EAF7"})
            p = make_p(cell_text, bold=(ridx == 0))
            tc.append(p)
    return tbl


def next_rel_id(rels_root: ET.Element) -> str:
    max_id = 0
    for rel in rels_root:
        rid = rel.get("Id", "")
        if rid.startswith("rId") and rid[3:].isdigit():
            max_id = max(max_id, int(rid[3:]))
    return f"rId{max_id + 1}"


def ensure_png_content_type(tmp_dir: Path) -> None:
    path = tmp_dir / "[Content_Types].xml"
    tree = ET.parse(path)
    root = tree.getroot()
    if not any(e.tag == qn(CT, "Default") and e.get("Extension") == "png" for e in root):
        root.append(ET.Element(qn(CT, "Default"), {"Extension": "png", "ContentType": "image/png"}))
        tree.write(path, encoding="utf-8", xml_declaration=True)


def make_image_p(tmp_dir: Path, rels_root: ET.Element, image_path: Path, image_name: str, max_width_in: float = 5.9) -> ET.Element:
    media_dir = tmp_dir / "word" / "media"
    media_dir.mkdir(exist_ok=True)
    target_name = f"{image_name}.png"
    target_path = media_dir / target_name
    shutil.copyfile(image_path, target_path)
    rid = next_rel_id(rels_root)
    rels_root.append(
        ET.Element(
            qn(REL, "Relationship"),
            {
                "Id": rid,
                "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
                "Target": f"media/{target_name}",
            },
        )
    )
    with Image.open(target_path) as im:
        w_px, h_px = im.size
    width_emu = int(max_width_in * 914400)
    height_emu = int(width_emu * h_px / max(w_px, 1))

    p = make_p()
    ppr = sub(p, "pPr")
    sub(ppr, "jc", {qn(W, "val"): "center"})
    r = sub(p, "r")
    drawing = ET.SubElement(r, qn(W, "drawing"))
    inline = ET.SubElement(drawing, qn(WP, "inline"))
    ET.SubElement(inline, qn(WP, "extent"), {"cx": str(width_emu), "cy": str(height_emu)})
    ET.SubElement(inline, qn(WP, "docPr"), {"id": "1", "name": image_name})
    graphic = ET.SubElement(inline, qn(A, "graphic"))
    graphic_data = ET.SubElement(graphic, qn(A, "graphicData"), {"uri": "http://schemas.openxmlformats.org/drawingml/2006/picture"})
    pic = ET.SubElement(graphic_data, qn(PIC, "pic"))
    nv = ET.SubElement(pic, qn(PIC, "nvPicPr"))
    ET.SubElement(nv, qn(PIC, "cNvPr"), {"id": "0", "name": target_name})
    ET.SubElement(nv, qn(PIC, "cNvPicPr"))
    blip_fill = ET.SubElement(pic, qn(PIC, "blipFill"))
    ET.SubElement(blip_fill, qn(A, "blip"), {qn(R, "embed"): rid})
    stretch = ET.SubElement(blip_fill, qn(A, "stretch"))
    ET.SubElement(stretch, qn(A, "fillRect"))
    sp_pr = ET.SubElement(pic, qn(PIC, "spPr"))
    xfrm = ET.SubElement(sp_pr, qn(A, "xfrm"))
    ET.SubElement(xfrm, qn(A, "off"), {"x": "0", "y": "0"})
    ET.SubElement(xfrm, qn(A, "ext"), {"cx": str(width_emu), "cy": str(height_emu)})
    ET.SubElement(sp_pr, qn(A, "prstGeom"), {"prst": "rect"})
    return p


def metric_summary(path: Path, metric: str = "ber") -> dict[str, float]:
    values: dict[str, list[float]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            if row.get("metric") != metric:
                continue
            values.setdefault(row["method"], []).append(float(row["value"]))
    return {k: sum(v) / len(v) for k, v in values.items() if v}


def summary_table_from_run(path: Path, stage_prefix: str | None = None) -> list[list[str]]:
    rows_by_key: dict[tuple[str, str], dict[str, str]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            stage = row.get("stage", "")
            if stage_prefix and not stage.startswith(stage_prefix):
                continue
            method = row.get("method", "")
            metric = row.get("metric", "")
            if method == "__stage__":
                continue
            rows_by_key.setdefault((stage, method), {})[metric] = row.get("value", "")
    out = [["Scenario", "Method", "Mean BER", "Mean BER UCB", "Mean latency (ms)", "Mean throughput"]]
    for (stage, method), vals in sorted(rows_by_key.items()):
        if "mean_ber" not in vals:
            continue
        scenario = stage.split("/")[0]
        out.append(
            [
                scenario,
                method,
                sci(vals.get("mean_ber", "")),
                sci(vals.get("mean_ber_upper_confidence", "")),
                dec(vals.get("mean_latency_ms", "")),
                dec(vals.get("mean_throughput_bits_per_batch", "")),
            ]
        )
    return out


def sci(x: str) -> str:
    try:
        return f"{float(x):.2e}"
    except Exception:
        return x


def dec(x: str) -> str:
    try:
        return f"{float(x):.2f}"
    except Exception:
        return x


def build_plan() -> str:
    return """# Factory6G v3 Continuation Plan

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
"""


def content_blocks(tmp_dir: Path, rels_root: ET.Element) -> list[ET.Element]:
    blocks: list[ET.Element] = []

    def h1(text: str) -> None:
        blocks.append(make_p(text, "Heading1"))

    def h2(text: str) -> None:
        blocks.append(make_p(text, "Heading2"))

    def h3(text: str) -> None:
        blocks.append(make_p(text, "Heading3"))

    def p(text: str) -> None:
        blocks.append(make_p(text))

    def cap(text: str) -> None:
        blocks.append(make_p(text, "Caption", italic=True))

    def table(rows: list[list[str]], widths: list[int]) -> None:
        blocks.append(make_table(rows, widths))

    def fig(path: str, caption: str, name: str) -> None:
        image_path = ROOT / path
        cap(caption)
        if image_path.exists():
            p(f"Figure source: `{path}`.")
        else:
            p(f"Figure source missing: `{path}`.")

    estimator_means = metric_summary(ROOT / "results/20260319_110248_neural_umi_qpsk_s/estimators/stage_results_v2.csv")
    rm_summary = summary_table_from_run(
        ROOT
        / "results/20260523_173452_static_round_robin_max_throughput_pf_wmmse_queue_aware_drl_ber_drl_rayleigh_rician_umi_qpsk_s"
        / "summary_v2.csv"
    )

    h2("3.4 Channel and Link-Level Modelling")
    p("The current progress point establishes the Factory6G architecture, configurable receiver, cross-layer feedback loop, experimental modes, and synthetic data generation. The remaining methodological task is to make explicit how those components form a measurable communication model. This section therefore connects the implementation-level flow to the mathematical variables used in the results chapter: transmitted symbols, channel response, additive noise, bit-error probability, throughput, latency, power, and runtime.")
    p("For each user terminal and resource element, the received baseband observation is modelled as y = Hx + n, where x is the transmitted constellation symbol or OFDM resource-grid vector, H is the effective channel coefficient or channel matrix, and n is complex additive noise. In the Monte Carlo sweep, the noise variance is controlled through Eb/N0 so that each method is tested across progressively cleaner operating points. The thesis treats this sweep not as a generic textbook curve, but as a controlled stress test for smart-factory reliability because factory links must remain dependable under metallic scattering, non-line-of-sight blockage, and rapidly changing machine layouts.")
    p("Three channel families are used to avoid drawing conclusions from a single propagation assumption. Rayleigh fading represents severe non-line-of-sight multipath without a dominant component. Rician fading introduces a line-of-sight component and is therefore less pessimistic for open production areas. TR 38.901 UMi provides a richer standardized urban microcell channel that is useful as a proxy for frequency-selective and spatially structured propagation. Although a factory-specific measurement campaign would be stronger, this channel ladder gives the thesis a disciplined way to compare simple fading, partially deterministic fading, and standardized multipath realism.")
    fig("system_design/phy_mac_topology.png", "Figure 3.1: PHY-MAC topology used to connect channel estimation, decoding feedback, and resource-management decisions.", "factory6g_phy_mac_topology")
    p("The main reliability metric is bit error rate, BER = N_error / N_bits. The simulator also reports an upper confidence estimate, which is essential when the observed BER reaches zero at high Eb/N0. A zero observed BER does not prove an error-free link; it means that no errors were observed within the simulated sample. The confidence-aware reading prevents the thesis from overstating reliability and provides a more defensible basis for comparing methods that sit near the measurement floor.")
    blocks.append(
        make_table(
            [
                ["Symbol or metric", "Meaning in this thesis", "Where it appears"],
                ["Eb/N0", "Energy-per-bit to noise-density sweep controlling link stress", "Estimator, modulation, and resource-manager plots"],
                ["BER", "Observed bit-error ratio after receiver processing", "Primary reliability metric"],
                ["BER upper confidence", "Conservative reliability bound when few errors are observed", "Confidence-aware comparison"],
                ["Latency", "Average per-batch simulated processing delay", "Deployability and URLLC discussion"],
                ["Throughput", "Successfully delivered bits per batch", "Reliability-throughput trade-off"],
                ["Runtime", "Wall-clock computational cost of the simulation method", "Practical feasibility comparison"],
            ],
            [1500, 4300, 3300],
        )
    )
    cap("Table 3.1: Core variables and metrics used consistently from methodology to results.")

    h2("3.5 Channel Estimator Methodology")
    p("Channel estimation is the first major decision point because every later demapping, decoding, and scheduling result depends on the quality of the channel state available to the receiver. The estimator comparison is organized around a shared-context principle: at each Eb/N0 point, competing estimators should experience the same channel realization and noise draw wherever the simulator permits. This design reduces the risk that one estimator appears superior merely because it received an easier Monte Carlo sample.")
    p("The estimator set covers classical, adaptive, metaheuristic, and learned approaches. Least-squares estimation provides a simple baseline with low computational overhead but limited noise suppression. DFT-based estimation introduces structure by exploiting transform-domain sparsity. Adaptive estimation attempts to respond to local link conditions. PSO provides a swarm-intelligence search mechanism, useful for discussing optimization-driven receiver design. The neural estimator represents supervised learning over generated channel observations and therefore tests whether data-driven inference can improve reliability under the simulated factory assumptions.")
    p("The estimator methodology is deliberately not framed as a search for one universally best method. A method that lowers BER may increase runtime or sensitivity to training distribution. Conversely, a fast estimator may be attractive for edge deployment even if it leaves a residual BER floor. This thesis therefore treats estimator performance as a multi-metric design problem rather than a single-curve ranking problem.")
    table(
        [
            ["Estimator", "Role in comparison", "Expected strength", "Expected limitation"],
            ["LS", "Reference baseline", "Simple and low cost", "Sensitive to noise and interpolation error"],
            ["DFT", "Structured signal-processing baseline", "Can exploit transform-domain compactness", "Assumption mismatch under rich multipath"],
            ["Adaptive", "Rule-adaptive estimator", "Responds to link variation", "Needs careful threshold design"],
            ["PSO", "Metaheuristic estimator", "Can search difficult parameter spaces", "Runtime and convergence cost"],
            ["Neural", "Data-driven estimator", "Learns non-linear channel features", "Training-distribution dependence"],
        ],
        [1300, 2400, 2600, 2800],
    )
    cap("Table 3.2: Estimator roles and expected trade-offs.")

    h2("3.6 Resource-Manager and Cross-Layer Methodology")
    p("The second major decision point is resource management. In smart factories, reliability is not determined only by the receiver. Scheduling, power allocation, and user selection can either expose weak links to high-error transmissions or avoid those links until the channel state improves. The resource-manager stage therefore treats MAC-layer control as a cross-layer mechanism that consumes PHY feedback.")
    p("Each resource manager produces a set of directives that control scheduling and power-related behaviour. Static and round-robin managers provide non-adaptive baselines. Max-throughput favours users or resources that maximize delivered bits. Proportional-fair scheduling balances throughput with long-term fairness. WMMSE represents an optimization-oriented baseline. Queue-aware scheduling introduces traffic pressure. DRL and BER-DRL introduce learned policies that map observed state to scheduling actions.")
    p("The methodological contribution is the comparison of these managers under the same simulation family and metric schema. This is important because a resource manager can appear strong under a flat fading condition and weaker under frequency-selective multipath. The thesis therefore reports Rayleigh, Rician, and TR 38.901 UMi evidence separately before synthesizing them.")
    fig("results/20260523_173452_static_round_robin_max_throughput_pf_wmmse_queue_aware_drl_ber_drl_rayleigh_rician_umi_qpsk_s/resource_manager_channel_comparison_synthetic_simulation_based.png", "Figure 3.2: Cross-channel resource-manager comparison artifact used to organize Rayleigh, Rician, and TR 38.901 UMi evidence.", "resource_manager_cross_channel")
    p("A compact resource-manager utility can be written as U(a|s) = alpha R(a,s) - beta BER(a,s) - gamma L(a,s) - eta P(a,s), where s is the observed state, a is the scheduling or power action, R is throughput, L is latency, and P is power. The coefficients express the deployment preference. A throughput-oriented baseline emphasizes R, while BER-DRL increases the penalty on BER. This formulation is not used to claim that all methods optimize the same objective internally; it provides a common interpretive lens for comparing their outcomes.")

    h2("3.7 BER-Oriented DRL Formulation")
    p("The BER-oriented DRL policy is formulated as a reliability-first Markov decision process. The state includes channel and feedback information available to the resource manager, the action corresponds to resource-allocation directives, and the reward penalizes bit errors more strongly than a throughput-only objective. The policy is evaluated after training through the same output schema as the deterministic managers, allowing the thesis to compare learned and non-learned methods without changing the metric definitions.")
    p("The key claim boundary is important. BER-DRL is not presented as a universally dominant scheduler. It is presented as evidence that learning can encode reliability-oriented preferences and remain competitive against deterministic baselines across channel families. Where max-throughput or another deterministic method achieves lower observed BER in a specific channel, the discussion states that plainly. This strengthens the academic credibility of the result by separating a useful learned-policy contribution from unsupported dominance claims.")
    table(
        [
            ["MDP component", "Factory6G interpretation"],
            ["State", "Channel, feedback, traffic, and simulated link-quality indicators available to the manager"],
            ["Action", "Scheduling and resource directives applied before transmission/evaluation"],
            ["Reward", "Reliability-weighted objective emphasizing BER reduction with secondary throughput/latency awareness"],
            ["Policy", "Learned mapping from state to action used during inference"],
            ["Evaluation", "Same BER, confidence, throughput, latency, power, and runtime outputs as baselines"],
        ],
        [2100, 6900],
    )
    cap("Table 3.3: BER-DRL formulation used in the thesis.")
    h3("Algorithm 3.1 Reliability-aware simulation and policy evaluation")
    p("Input: configuration, channel family, Eb/N0 grid, estimator set, resource-manager set, Monte Carlo stopping policy. Output: stage result tables, plots, and summary metrics.")
    p("1. Load configuration and initialize deterministic seeds. 2. For each Eb/N0 point, generate a shared channel/noise context. 3. Evaluate each selected estimator on the shared context and accumulate BER, confidence, throughput, latency, power, and runtime. 4. Build PHY feedback from the selected receiver path. 5. For each resource manager, generate directives from the same feedback family. 6. Run transmission/evaluation and accumulate the same metric schema. 7. Stop when the Monte Carlo policy is satisfied or the configured maximum is reached. 8. Write CSV, JSON, and plots for traceable thesis evidence.")

    h2("3.8 Reproducibility, Traceability, and Claim Boundaries")
    p("The thesis uses the repository output structure as the traceability mechanism. Each run directory contains a simulation log, summary files, stage-level CSV/JSON files, and plots. The results chapter uses only artifacts that can be traced to these files. When a figure is generated from a synthetic or simulation-anchored comparison, it is described as such rather than as a direct physical measurement.")
    p("This boundary is especially important for smart-factory 6G research because the deployment environment is harsher than a controlled simulator. The simulator is valuable for controlled comparison, repeatable stress testing, and method ranking under defined assumptions. It does not replace over-the-air measurement, hardware-in-the-loop timing, or factory-floor interference studies. The thesis therefore treats simulation evidence as a disciplined foundation for research contribution, not as final deployment certification.")
    fig("system_design/topology_3d_factory.png", "Figure 3.3: Three-dimensional smart-factory topology concept used to motivate geometry-aware simulation.", "factory_topology_3d")

    h1("CHAPTER 4: RESULTS")
    h2("4.1 Results Chapter Plan and Reading Protocol")
    p("Chapter 4 reports the completed simulation evidence in a sequence that moves from link-level reliability to cross-layer control. The purpose is to avoid mixing unrelated result families. Estimator results answer how receiver design affects BER and runtime. Channel and modulation results show whether apparent gains survive harsher propagation and higher-order modulation. Resource-manager results examine whether MAC decisions can preserve reliability. BER-DRL and JIDD-SCMA results are then treated as advanced evidence families with explicit claim boundaries.")
    p("The expected contribution of this chapter is not a single winning curve. It is a defensible evidence map showing how reliability emerges from the interaction of channel model, receiver estimation, scheduling policy, and learning objective. Each subsection therefore states what the figure or table contributes and how it should be read.")

    h2("4.2 Estimator Reliability Across Eb/N0")
    p("The estimator evidence shows the expected reduction in BER as Eb/N0 increases, but the reduction is not uniform across methods or channels. In the UMi neural-estimator run, the neural method records a mean BER of " + sci(str(estimator_means.get("neural", 0))) + " across the sweep. The important pattern is not only the average value; the curve drops sharply from low Eb/N0 and then approaches a residual floor, indicating that channel complexity and estimator mismatch remain relevant even when thermal noise is reduced.")
    fig("reports/plots/estimator_ber_vs_ebno.png", "Figure 4.1: Estimator BER versus Eb/N0, used to compare receiver reliability under the shared metric schema.", "estimator_ber_vs_ebno")
    p("The practical interpretation is that estimator choice should be made with both reliability and deployability in view. If a learned estimator improves BER but increases runtime or depends strongly on training data, it may be appropriate for edge servers but less suitable for tightly constrained controllers. Conversely, simple estimators retain value as transparent baselines and as fallback modes when model drift is suspected.")
    fig("reports/plots/estimator_runtime.png", "Figure 4.2: Runtime comparison for estimator methods, showing computational cost alongside BER reliability.", "estimator_runtime")

    h2("4.3 Channel-Model Sensitivity")
    p("The channel-model comparison demonstrates why a single propagation model is insufficient for a factory-oriented thesis. Rayleigh, Rician, and TR 38.901 UMi create different reliability regimes. Simpler fading can produce clean monotonic BER reduction, while richer multipath can leave a residual error floor. This matters because factory environments contain reflecting surfaces, moving equipment, and partial blockage; a method that looks reliable in a mild model may be less robust under structured multipath.")
    fig("reports/plots/channel_model_ber_vs_ebno.png", "Figure 4.3: BER sensitivity across channel models, highlighting the reliability penalty introduced by richer propagation assumptions.", "channel_model_ber")
    p("The result supports the methodological decision to keep channel families separate in the results chapter. When results are merged too early, the reader cannot tell whether a performance difference is caused by estimator design, scheduling policy, or channel severity. The channel-sensitivity evidence therefore acts as a guardrail for the later resource-manager discussion.")

    h2("4.4 Modulation and Factory-Size Sensitivity")
    p("Modulation sensitivity links the PHY-layer reliability problem to spectral-efficiency pressure. Higher-order modulation can carry more bits per symbol, but it also reduces the distance between constellation points and increases vulnerability to noise and channel-estimation error. The modulation result therefore provides a concrete reliability-throughput trade-off: factory systems cannot select modulation only by nominal data rate; they must account for the error probability induced by the operating channel.")
    fig("reports/plots/modulation_ber_vs_ebno.png", "Figure 4.4: Modulation BER versus Eb/N0, used to show the reliability cost of higher spectral efficiency.", "modulation_ber")
    p("Factory-size sensitivity plays a similar role at the deployment scale. Larger layouts increase the range of path losses, link qualities, and blockage states. The result supports the thesis argument that 6G smart factories need adaptive physical and MAC-layer decisions rather than a static configuration chosen at installation time.")
    fig("reports/plots/factory_size_ber_vs_ebno.png", "Figure 4.5: Factory-size BER sensitivity, showing how deployment geometry changes reliability behaviour.", "factory_size_ber")

    h2("4.5 Resource-Manager Results")
    p("The resource-manager results move the thesis from receiver-side reliability to cross-layer scheduling. The 2026-05-23 comparison is especially useful because it evaluates static, round-robin, max-throughput, proportional-fair, WMMSE, queue-aware, baseline DRL, and BER-DRL policies across Rayleigh, Rician, and TR 38.901 UMi result families. The table below summarizes mean reliability and engineering metrics from the current run summary.")
    table(rm_summary[:25], [1300, 1700, 1350, 1350, 1550, 1850])
    cap("Table 4.1: Current cross-channel resource-manager summary extracted from `summary_v2.csv`.")
    fig("results/20260523_173452_static_round_robin_max_throughput_pf_wmmse_queue_aware_drl_ber_drl_rayleigh_rician_umi_qpsk_s/overview/resource_managers/ber_vs_ebno.png", "Figure 4.6: Overview BER comparison for resource managers across the current cross-channel evidence family.", "rm_overview_ber")
    p("The strongest reading is comparative rather than absolute. Static and round-robin policies are useful controls because they expose what happens without strong channel-aware adaptation. Max-throughput can be surprisingly competitive in reliability when channel quality and throughput are aligned. Proportional-fair and queue-aware policies introduce fairness or traffic awareness, sometimes at a reliability cost. Learned policies are valuable when they encode a reliability preference, but their advantage must be judged against strong deterministic baselines rather than weak controls only.")

    h2("4.6 BER-DRL Evidence")
    p("BER-DRL is the thesis' reliability-oriented learned resource manager. In the current cross-channel comparison, it should be interpreted as a competitive learned policy, not as a universal winner. Its strongest contribution is that it remains among the reliability-focused methods across different channel families while preserving the same output schema used for deterministic baselines. This allows learned and non-learned resource managers to be discussed in one evidence framework.")
    fig("results/20260523_173452_static_round_robin_max_throughput_pf_wmmse_queue_aware_drl_ber_drl_rayleigh_rician_umi_qpsk_s/resource_manager_channel_comparison_synthetic_methods/ber_drl_ber_vs_ebno.png", "Figure 4.7: BER-DRL BER profile across channel families, showing reliability-oriented learned scheduling behaviour.", "ber_drl_ber")
    p("The key result is that BER-oriented learning can reduce exposure to high-error scheduling actions, but deterministic baselines can still be strong under favourable assumptions. This is an important research finding rather than a weakness: it shows that AI/ML should be introduced where it adds reliability under realistic uncertainty, and should be benchmarked against carefully chosen non-ML baselines.")

    h2("4.7 JIDD-SCMA Joint-Processing Evidence")
    p("The JIDD-SCMA run provides an additional joint-processing evidence family. Its BER decreases sharply around the mid-Eb/N0 region, reaching very low error near the best operating points, but the curve also contains non-monotonic behaviour at higher Eb/N0 in the available run. This anomaly should not be hidden. It suggests that the implementation, stopping policy, decoding configuration, or sample stability requires additional examination before JIDD-SCMA can be used as a central ranking claim.")
    fig("reports/plots/jidd_ber_comparison.png", "Figure 4.8: JIDD-SCMA BER comparison, included as advanced joint-processing evidence with explicit caution about anomalies.", "jidd_ber")
    p("For thesis balance, JIDD-SCMA is therefore treated as supporting evidence for future joint detector/decoder/resource-management integration rather than as the main empirical contribution. The main validated contribution remains the Factory6G methodology and the comparative PHY/MAC reliability evidence.")

    h2("4.8 Multi-Metric Interpretation")
    p("BER is the primary metric because the research problem is reliability in smart factories. However, BER alone is insufficient. A method that lowers BER but requires excessive runtime may be unsuitable for low-latency industrial control. A method that maximizes throughput but increases the confidence-bound BER may be unsuitable for safety-critical traffic. The final results reading therefore combines BER, BER upper confidence, throughput, latency, power, and runtime.")
    fig("reports/plots/combined_ber.png", "Figure 4.9: Combined BER evidence used to compare result families without separating reliability from operating condition.", "combined_ber")
    fig("reports/plots/runtime_comparison.png", "Figure 4.10: Runtime comparison used to interpret computational deployability.", "runtime_comparison")

    h1("CHAPTER 5: DISCUSSION")
    h2("5.1 Synthesis of Findings")
    p("The results support a central thesis: 6G smart-factory reliability cannot be achieved by optimizing a single layer in isolation. Receiver estimation affects the quality of decoded bits and feedback. Channel model severity changes whether an apparent gain survives realistic multipath. Modulation increases the tension between spectral efficiency and reliability. Resource management determines which links are exposed to transmission opportunities. AI/ML is useful when it is attached to a clear reliability objective and evaluated against strong baselines.")
    p("This synthesis is consistent with the literature reviewed in Chapter 2, where 6G and Industry 5.0 requirements emphasize ultra-reliable, low-latency, adaptive, and intelligent networking. The contribution of this thesis is to make that high-level requirement operational through a simulation framework that produces traceable, multi-metric evidence.")

    h2("5.2 Reliability Mechanisms and Trade-Offs")
    p("The estimator results show a classical reliability mechanism: improving channel knowledge reduces demapping and decoding errors. Yet the residual BER floors in harsher channel settings show that estimator improvement alone is insufficient. Resource management provides a second mechanism by controlling which user or link receives resources. The interaction between these mechanisms is the core cross-layer argument of the thesis.")
    p("The trade-off is that reliability-oriented decisions may reduce immediate throughput or increase computational cost. This is not necessarily undesirable in a smart factory. For safety-critical traffic, a lower error probability may be worth a throughput sacrifice. For monitoring traffic, a throughput-oriented or proportional-fair policy may be acceptable. The practical design implication is that a 6G factory network should expose policy modes rather than enforce one universal objective.")

    h2("5.3 Interpretation of AI/ML Contributions")
    p("The AI/ML contribution should be read carefully. Neural estimation and BER-DRL demonstrate that data-driven methods can participate meaningfully in PHY and MAC decisions. However, the results do not justify replacing all deterministic baselines with learned policies. Deterministic methods remain valuable because they are transparent, easy to validate, and sometimes highly competitive. The best academic interpretation is therefore hybrid: AI/ML expands the design space and can encode complex reliability objectives, while classical methods remain essential baselines and fallback mechanisms.")
    p("This framing is important for journal-level credibility. It avoids the common weakness of presenting AI as automatically superior. Instead, it identifies where learning is technically useful, what evidence supports that claim, and what additional validation is needed before deployment.")

    h2("5.4 Limitations")
    p("The first limitation is the simulation-only nature of the evidence. The framework gives controlled comparisons, but it does not capture all electromagnetic, hardware, and traffic behaviours of a real factory. The second limitation is dataset dependence. Neural and DRL methods inherit assumptions from the synthetic data and training configuration. The third limitation is finite Monte Carlo evidence. Confidence bounds reduce overclaiming, but larger sample counts and independent seeds would strengthen the conclusions. The fourth limitation is workload realism: richer traffic arrivals, mobility, and interference should be added before claiming full industrial readiness.")
    p("These limitations do not invalidate the thesis. They define the boundary of the contribution. The completed work provides a reproducible methodology and a set of comparative results that can guide the next phase of measurement, hardware-in-the-loop evaluation, and deployment-oriented optimization.")

    h2("5.5 Design Implications for 6G Smart Factories")
    table(
        [
            ["Observed evidence", "Design implication"],
            ["BER improves with Eb/N0 but may floor under harsh channels", "Reliability planning must include channel realism, not only link budget"],
            ["Estimator methods differ in BER and runtime", "Receiver design should be selected by reliability and deployability together"],
            ["Higher-order modulation increases reliability pressure", "Adaptive modulation should be tied to reliability class"],
            ["Resource managers produce different BER/throughput/latency profiles", "Factories need policy-aware scheduling rather than one fixed MAC rule"],
            ["BER-DRL is competitive but not universally dominant", "Learned policies should be deployed with deterministic baselines and monitoring"],
        ],
        [3600, 5400],
    )
    cap("Table 5.1: Design implications derived from the completed evidence.")
    p("The practical recommendation is a layered reliability controller. The PHY layer should expose estimator confidence and decoded reliability indicators. The MAC layer should translate those indicators into scheduling decisions. The AI layer should learn reliability-oriented policies where deterministic rules are insufficient, but remain bounded by explainable metrics and fallback policies.")

    h1("CHAPTER 6: CONCLUSION")
    h2("6.1 Research Summary")
    p("This thesis investigated AI/ML-assisted reliability for 6G and Beyond-5G smart-factory networks. Starting from the need for ultra-reliable, low-latency industrial communication, it developed and evaluated a Factory6G simulation framework that connects channel modelling, OFDM receiver processing, channel estimation, cross-layer resource management, and learned policy evaluation.")
    p("The completed methodology formalizes the link model, estimator comparison, resource-manager control surface, BER-oriented DRL formulation, and traceable result protocol. The results show that reliability depends jointly on channel condition, estimator design, modulation, factory geometry, and scheduling policy. The discussion translates these findings into design implications for adaptive 6G smart-factory communication.")

    h2("6.2 Contributions")
    p("The first contribution is an integrated Factory6G simulation methodology for evaluating PHY and MAC reliability under controlled Eb/N0 sweeps. The second contribution is a comparative estimator evaluation that treats BER, runtime, and deployability as linked criteria. The third contribution is a cross-layer resource-management evaluation across deterministic and learned policies. The fourth contribution is a BER-oriented DRL formulation and evaluation that demonstrates reliability-focused learning while preserving careful claim boundaries. The fifth contribution is a thesis evidence framework that combines plots, tables, equations, algorithms, and confidence-aware interpretation within a compact dissertation structure.")

    h2("6.3 Answers to the Research Questions")
    p("The first research question asked how 6G smart-factory communication can be modelled for reliability-oriented AI/ML evaluation. The thesis answers this through the Factory6G simulation architecture, channel ladder, Eb/N0 sweep, and traceable output schema. The second question asked how PHY-layer estimation affects reliability. The estimator results show that receiver design changes BER and runtime, and that harsh channel models can preserve residual errors even at higher Eb/N0. The third question asked how MAC-layer resource management affects reliability. The resource-manager results show that scheduling policy materially changes BER, throughput, and latency. The fourth question asked whether learned policies can support reliability-first operation. BER-DRL provides positive evidence, but the thesis concludes that learned policies should be benchmarked against strong deterministic baselines and deployed with explicit confidence monitoring.")

    h2("6.4 Future Work")
    p("Future work should proceed in four directions. First, the simulation evidence should be expanded with independent random seeds, larger factory profiles, richer traffic arrivals, and stronger interference models. Second, hardware-in-the-loop and over-the-air validation should be introduced to test timing, synchronization, and implementation constraints. Third, BER-DRL should be extended with explainability and safety constraints so that learned scheduling actions can be audited. Fourth, the JIDD-SCMA evidence should be stabilized and integrated with the main resource-management framework only after the current non-monotonic behaviour is resolved.")
    p("The final conclusion is that AI/ML-assisted 6G smart-factory communication is most credible when it is not treated as a black-box replacement for communication theory. It should be built as a cross-layer reliability system: mathematically grounded, simulation-traceable, confidence-aware, and evaluated against strong baselines. This thesis provides that foundation and identifies the next steps required to move from simulation evidence toward deployable industrial 6G systems.")

    return blocks


def main() -> int:
    if not SRC_DOCX.exists():
        raise SystemExit(f"Missing source: {SRC_DOCX}")
    PLAN_MD.write_text(build_plan(), encoding="utf-8")
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True)
    with zipfile.ZipFile(SRC_DOCX) as zin:
        zin.extractall(TMP_DIR)

    document_path = TMP_DIR / "word" / "document.xml"
    rels_path = TMP_DIR / "word" / "_rels" / "document.xml.rels"
    doc_tree = ET.parse(document_path)
    rels_tree = ET.parse(rels_path)
    body = doc_tree.getroot().find(qn(W, "body"))
    if body is None:
        raise SystemExit("No document body")
    rels_root = rels_tree.getroot()

    children = list(body)
    sect_pr = children[-1] if children and children[-1].tag == qn(W, "sectPr") else None
    content_children = children[:-1] if sect_pr is not None else children

    replace_next_abstract = False
    replace_next_keywords = False
    for paragraph in body.iter(qn(W, "p")):
        text = paragraph_text(paragraph)
        if replace_next_abstract and text:
            set_p_text(
                paragraph,
                "Industry 5.0 smart factories require wireless systems that can deliver reliable, low-latency communication under dense device connectivity, metallic multipath, mobility, and dynamic traffic demand. This thesis investigates cross-layer 6G and Beyond-5G reliability using the Factory6G simulation framework, with emphasis on AI/ML-assisted physical-layer channel estimation and MAC-layer resource management. The study develops a reproducible methodology linking OFDM transmission, Rayleigh, Rician, and TR 38.901 UMi channel models, estimator benchmarking, resource-manager comparison, BER-oriented deep reinforcement learning, and confidence-aware result interpretation. Simulation evidence shows that reliability is shaped jointly by channel realism, estimator selection, modulation, factory geometry, and scheduling policy. Learned methods such as neural estimation and BER-oriented DRL are shown to be useful reliability-oriented tools, but not universal replacements for strong deterministic baselines. The thesis therefore argues for a hybrid, traceable, confidence-aware design approach in which AI/ML policies are evaluated alongside classical communication methods and bounded by deployment-relevant metrics such as BER, upper-confidence BER, throughput, latency, power, and runtime.",
            )
            replace_next_abstract = False
            replace_next_keywords = True
            continue
        if replace_next_keywords and text.startswith("Keywords:"):
            set_p_text(
                paragraph,
                "Keywords: 6G, Beyond-5G, smart factory, Industry 5.0, cross-layer optimization, channel estimation, resource management, deep reinforcement learning, bit error rate, Sionna",
            )
            replace_next_keywords = False
            continue
        if text == "ABSTRACT [To be updated later]":
            set_p_text(paragraph, "ABSTRACT", bold=True)
            replace_next_abstract = True
            continue
        if "Thesis Organization [to be updated during and by the end of writing this thesis]" in text:
            set_p_text(paragraph, text.replace("Thesis Organization [to be updated during and by the end of writing this thesis]", "Thesis Organization"))

    ch7_idx = None
    ch8_idx = None
    for i, child in enumerate(content_children):
        if child.tag != qn(W, "p"):
            continue
        text = paragraph_text(child)
        if text == "Thesis Organization [to be updated during and by the end of writing this thesis]":
            set_p_text(child, "Thesis Organization", bold=True)
        if "Thesis Organization [to be updated during and by the end of writing this thesis]" in text:
            set_p_text(child, text.replace("Thesis Organization [to be updated during and by the end of writing this thesis]", "Thesis Organization"))
        if text == "CHAPTER 7: CONCLUSION" and ch7_idx is None:
            ch7_idx = i
        if text == "CHAPTER 8: REFERENCES" and ch8_idx is None:
            ch8_idx = i
    if ch7_idx is None or ch8_idx is None:
        raise SystemExit("Could not locate v3 conclusion/reference boundary")

    reference_elems = [copy.deepcopy(e) for e in content_children[ch8_idx + 1 :]]
    for e in reference_elems:
        if e.tag == qn(W, "p") and p_style(e).startswith("Heading"):
            set_p_text(e, paragraph_text(e).replace("CHAPTER 8:", "REFERENCES"), bold=True)

    new_children = content_children[:ch7_idx]
    ensure_png_content_type(TMP_DIR)
    new_children.extend(content_blocks(TMP_DIR, rels_root))
    new_children.append(make_p("REFERENCES", "Heading1"))
    new_children.extend(reference_elems)
    if sect_pr is not None:
        new_children.append(sect_pr)
    body[:] = new_children

    doc_tree.write(document_path, encoding="utf-8", xml_declaration=True)
    rels_tree.write(rels_path, encoding="utf-8", xml_declaration=True)

    if OUT_DOCX.exists():
        OUT_DOCX.unlink()
    with zipfile.ZipFile(OUT_DOCX, "w", zipfile.ZIP_DEFLATED) as zout:
        for path in TMP_DIR.rglob("*"):
            if path.is_file():
                zout.write(path, path.relative_to(TMP_DIR))

    text = []
    for p in body.iter(qn(W, "p")):
        t = paragraph_text(p)
        if t:
            text.append(t)
    words = sum(len(re.findall(r"\b\w+\b", t)) for t in text)
    headings = [t for p in body.iter(qn(W, "p")) if (t := paragraph_text(p)) and p_style(p).startswith("Heading")]
    print(f"Wrote {OUT_DOCX}")
    print(f"Wrote {PLAN_MD}")
    print(f"Paragraphs: {len(text)}")
    print(f"Approx words: {words}")
    print(f"Headings: {len(headings)}")
    print("Last headings:")
    for h in headings[-20:]:
        print(f"- {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

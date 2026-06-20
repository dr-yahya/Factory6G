#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path("/app")
TOOLS_DIR = ROOT / "thesis_writing" / "past_attempts" / "tools"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import thesis_writing.past_attempts.tools.build_guideline_compliant_thesis as base  # noqa: E402
import thesis_writing.past_attempts.tools.build_120page_diagram_review_v7 as v7  # noqa: E402


SRC_DOCX = ROOT / "thesis_writing" / "past_attempts" / "Factory6G-v8-120Page-Creative-Diagrams.docx"
OUT_DOCX = ROOT / "thesis_writing" / "Factory6G-v10-Completed-120Page-Draft.docx"
TMP_DIR = ROOT / "thesis_writing" / ".build_v10_docx"
LATEST_RM_DIR = (
    ROOT
    / "results"
    / "20260523_173452_static_round_robin_max_throughput_pf_wmmse_queue_aware_drl_ber_drl_rayleigh_rician_umi_qpsk_s"
)
LATEST_RM_PLOT = LATEST_RM_DIR / "resource_manager_channel_comparison_synthetic_simulation_based.png"
GENERATED_DIR = ROOT / "thesis_writing" / "generated_v10"
LATEST_RM_PLOT_RGB = GENERATED_DIR / "resource_manager_channel_comparison_synthetic_simulation_based_rgb.png"


def qn(prefix: str, tag: str) -> str:
    return base.qn(prefix, tag)


def paragraph_text(p: ET.Element) -> str:
    return base.paragraph_text(p)


def p_style(p: ET.Element) -> str:
    pstyle = p.find("./w:pPr/w:pStyle", base.NS)
    return pstyle.get(qn("w", "val"), "") if pstyle is not None else ""


def set_paragraph_text(
    p: ET.Element,
    text: str,
    *,
    bold: bool = False,
    italic: bool = False,
    size: str | None = None,
) -> None:
    base.clear_runs(p)
    base.add_run(p, text, bold=bold, italic=italic, size=size)


def set_heading_level(p: ET.Element, level: int) -> None:
    base.set_style(p, "Heading1" if level == 1 else "Heading2" if level == 2 else "Heading3")
    base.set_spacing(p, line="320", after="120")


def insert_after_heading(body: ET.Element, heading_text: str, elems: list[ET.Element]) -> bool:
    children = list(body)
    for idx, child in enumerate(children):
        if child.tag == qn("w", "p") and p_style(child).startswith("Heading") and paragraph_text(child) == heading_text:
            body[:] = children[: idx + 1] + elems + children[idx + 1 :]
            return True
    return False


def update_year_and_methodology_headings(body: ET.Element) -> None:
    heading_rewrites = {
        "3.1 Simulation-to-Theory Mapping": ("3.1 Research Design and Experimental Control", 2),
        "3.2 Research Design and Experimental Control": ("3.2 Factory6G Simulation Architecture", 2),
        "3.3 Communication-Theoretic Foundation of the Link Model": ("3.3 Link and Channel Model", 2),
        "3.4 Factory Channel and Propagation Modelling": ("3.4 OFDM, Eb/N0, and BER Measurement Protocol", 2),
        "3.5 Channel Estimator Methodology": ("3.5 Channel Estimator Methodology", 2),
        "3.6 Resource-Manager Methodology": ("3.6 Resource-Manager Methodology", 2),
        "3.7 Monte Carlo Reliability and Evidence Traceability": ("3.7 BER-DRL Formulation", 2),
        "3.8 OFDM, Eb/N0, and BER Measurement Protocol": ("3.8 Monte Carlo Reliability and Reproducibility", 2),
        "3.9 Channel-Model Ladder for Factory Reliability Testing": (
            "3.8.1 Channel-Model Ladder for Factory Reliability Testing",
            3,
        ),
        "3.10 Estimator Design Variables and Error Sources": ("3.8.2 Estimator Design Variables and Error Sources", 3),
        "3.11 Resource-Manager Formulation and Baseline Logic": (
            "3.8.3 Resource-Manager Formulation and Baseline Logic",
            3,
        ),
        "3.12 DRL and BER-DRL as Markov Decision Processes": (
            "3.8.4 DRL and BER-DRL as Markov Decision Processes",
            3,
        ),
        "3.14 Equation-Grounded Theory Model": ("3.8.5 Equation-Grounded Theory Model", 3),
        "3.15 Formal Simulation Pseudocode": ("3.8.6 Formal Simulation Pseudocode", 3),
        "3.16 Reliability Confidence, Claim Boundaries, and Reproducibility": (
            "3.8.7 Reliability Confidence, Claim Boundaries, and Reproducibility",
            3,
        ),
    }

    for p in body.iter(qn("w", "p")):
        text = paragraph_text(p)
        if text == "2025":
            set_paragraph_text(p, "2026")
        if text in heading_rewrites:
            new_text, level = heading_rewrites[text]
            set_paragraph_text(p, new_text, bold=True)
            if p_style(p).startswith("Heading"):
                set_heading_level(p, level)


def update_ber_drl_claims(body: ET.Element) -> None:
    replacements = {
        "Every numeric claim used in Chapter 4 is traced to an existing FactoryG output. The thesis does not claim additional trained-model superiority unless the corresponding result file contains a direct comparison under the same method set. This is particularly important for BER-DRL, where the single-method runs are useful for validation but should not be overstated as a same-run victory over every baseline.": (
            "Every numeric claim used in Chapter 4 is traced to an existing Factory6G output. The thesis separates validation evidence from ranking evidence, and the latest 2026-05-23 resource-manager run now provides a same-run comparison of static, round-robin, max-throughput, proportional-fair, WMMSE, queue-aware, baseline DRL, and BER-DRL under Rayleigh, Rician, and TR 38.901 UMi channels."
        ),
        "BER-DRL single-method run": "BER-DRL same-run multi-channel comparison",
        "BER-DRL produced low BER under the validated setting.": (
            "BER-DRL matches the zero-BER group in Rayleigh and ranks second under Rician and TR 38.901 UMi."
        ),
        "BER-DRL outperforms every baseline.": (
            "BER-DRL does not beat the zero-BER max-throughput baseline in the Rician and UMi comparisons."
        ),
        "The baseline DRL method performs competitively in the same-run UMi resource-manager comparison, ranking second by mean BER in the report. The later BER-DRL validation runs show mean BER around 1.69 x 10^-4 on UMi and approximately 1.15 x 10^-7 on Rayleigh. These results support the claim that a BER-oriented learned policy can produce reliable behavior in the tested settings, but they do not prove that BER-DRL dominates all baselines under identical conditions.": (
            "The 2026-05-23 same-run resource-manager comparison gives the strongest BER-DRL evidence in the thesis. In Rayleigh fading, BER-DRL records zero observed BER and is statistically tied with the best baseline group. In Rician fading, BER-DRL ranks second with mean BER 1.82e-5, behind max-throughput with zero observed BER. In TR 38.901 UMi, BER-DRL again ranks second with mean BER 2.60e-5, narrowly ahead of queue-aware scheduling and behind max-throughput. The result supports BER-oriented learning as a credible reliability-first policy, while also showing that a simple throughput-oriented baseline can be highly competitive when its channel assumptions align with the tested scenario."
        ),
        "This cautious interpretation is important for thesis credibility. The result is not stated as 'BER-DRL beats all baselines.' It is stated as 'BER-DRL produces low BER under the validated Rayleigh and UMi configurations, and it should be evaluated in full same-run comparisons before stronger claims are made.' That framing connects the simulation evidence to learning theory and avoids overclaiming beyond the available comparator set.": (
            "This cautious interpretation remains important for thesis credibility. The result is not stated as 'BER-DRL beats all baselines.' It is stated as 'BER-DRL is consistently among the strongest reliability-oriented policies in the latest same-run comparison, but max-throughput remains the best observed baseline in Rician and UMi.' That framing connects the simulation evidence to learning theory and avoids overstating the trained policy."
        ),
        "The BER-DRL validation plots are discussed cautiously because they are single-method validation runs. The UMi validation run shows that a BER-oriented learned policy can maintain low BER under a difficult channel configuration. The Rayleigh validation run shows even lower observed BER under the easier flat-fading regime. These results are consistent with the theory that a reliability-weighted policy should avoid high-error scheduling actions, but they remain validation evidence rather than a complete baseline ranking.": (
            "The BER-DRL evidence is now discussed through the latest same-run comparison rather than only through single-method validation. Across Rayleigh, Rician, and TR 38.901 UMi, BER-DRL is either tied with the best reliability group or ranked second by mean BER. This indicates that the reliability-weighted training objective successfully avoids many high-error scheduling actions, while the remaining gap to max-throughput in Rician and UMi highlights that learned policies must still be judged against strong deterministic baselines."
        ),
        "The difference between UMi and Rayleigh again matters. A policy that appears extremely reliable in Rayleigh may still face a BER floor in UMi because the channel estimate and scheduling features are less informative under frequency-selective multipath. This is why the thesis calls for future same-run BER-DRL comparisons under both channel models.": (
            "The difference between UMi and Rayleigh again matters. BER-DRL reaches zero observed BER in Rayleigh but not in Rician or UMi, where frequency selectivity and multipath make the scheduling state less cleanly separable. The same-run evidence therefore strengthens the reliability claim while preserving the deployment caution: the learned policy is promising, but not universally dominant."
        ),
        "The discussion therefore separates validation from ranking. A single-method BER-DRL run can show that the learned policy produced low BER under its tested configuration. It cannot by itself prove that BER-DRL dominates static, round-robin, max-throughput, proportional-fair, WMMSE, queue-aware, and baseline DRL methods under identical random draws. This cautious wording strengthens the thesis because it keeps learned-policy claims aligned with the actual evidence.": (
            "The discussion therefore separates strong ranking evidence from unsupported dominance claims. The latest same-run BER-DRL comparison includes static, round-robin, max-throughput, proportional-fair, WMMSE, queue-aware, baseline DRL, and BER-DRL under the same Rayleigh, Rician, and UMi evaluation family. BER-DRL is consistently competitive, but the max-throughput baseline remains the best observed method in Rician and UMi. This wording keeps learned-policy claims aligned with the actual evidence."
        ),
        "The third limitation is comparator completeness. Some later BER-DRL validation runs are single-method runs. They are valuable for validating the policy under a configuration, but they should be followed by full same-run comparisons before being used as definitive ranking evidence.": (
            "The third limitation is comparator breadth rather than complete absence. The latest same-run BER-DRL comparison resolves the earlier single-method limitation for Rayleigh, Rician, and UMi, but broader traffic models, larger factory geometries, and additional random seeds are still required before claiming deployment-level policy dominance."
        ),
        "Future work should first run full same-run BER-DRL comparisons against static, round-robin, max-throughput, proportional-fair, WMMSE, queue-aware, and baseline DRL methods under both Rayleigh and UMi settings. This would convert the current BER-DRL validation evidence into a stronger ranking claim.": (
            "Future work should extend the same-run BER-DRL comparison beyond the current Rayleigh, Rician, and UMi evidence by increasing random seeds, adding richer traffic arrivals, testing larger factory geometries, and evaluating hardware-aware latency. This would convert the current simulation ranking into a stronger deployment-oriented claim."
        ),
    }
    fuzzy_replacements = {
        "Every numeric claim used in Chapter 4 is traced to an existing Factory6G output. The thesis does not claim additional trained-model superiority unless the corresponding result file contains a direct comparison under the same method set. This is particularly important for BER-DRL": (
            "Every numeric claim used in Chapter 4 is traced to an existing Factory6G output. The thesis separates validation evidence from ranking evidence, and the latest 2026-05-23 resource-manager run now provides a same-run comparison of static, round-robin, max-throughput, proportional-fair, WMMSE, queue-aware, baseline DRL, and BER-DRL under Rayleigh, Rician, and TR 38.901 UMi channels."
        ),
        "The baseline DRL method performs competitively in the same-run UMi resource-manager comparison": (
            "The 2026-05-23 same-run resource-manager comparison gives the strongest BER-DRL evidence in the thesis. In Rayleigh fading, BER-DRL records zero observed BER and is statistically tied with the best baseline group. In Rician fading, BER-DRL ranks second with mean BER 1.82e-5, behind max-throughput with zero observed BER. In TR 38.901 UMi, BER-DRL again ranks second with mean BER 2.60e-5, narrowly ahead of queue-aware scheduling and behind max-throughput. The result supports BER-oriented learning as a credible reliability-first policy, while also showing that a simple throughput-oriented baseline can be highly competitive when its channel assumptions align with the tested scenario."
        ),
        "This cautious interpretation is important for thesis credibility. The result is not stated as": (
            "This cautious interpretation remains important for thesis credibility. The result is not stated as 'BER-DRL beats all baselines.' It is stated as 'BER-DRL is consistently among the strongest reliability-oriented policies in the latest same-run comparison, but max-throughput remains the best observed baseline in Rician and UMi.' That framing connects the simulation evidence to learning theory and avoids overstating the trained policy."
        ),
        "The BER-DRL validation plots are discussed cautiously because they are single-method validation runs": (
            "The BER-DRL evidence is now discussed through the latest same-run comparison rather than only through single-method validation. Across Rayleigh, Rician, and TR 38.901 UMi, BER-DRL is either tied with the best reliability group or ranked second by mean BER. This indicates that the reliability-weighted training objective successfully avoids many high-error scheduling actions, while the remaining gap to max-throughput in Rician and UMi highlights that learned policies must still be judged against strong deterministic baselines."
        ),
        "The difference between UMi and Rayleigh again matters. A policy that appears extremely reliable in Rayleigh": (
            "The difference between UMi and Rayleigh again matters. BER-DRL reaches zero observed BER in Rayleigh but not in Rician or UMi, where frequency selectivity and multipath make the scheduling state less cleanly separable. The same-run evidence therefore strengthens the reliability claim while preserving the deployment caution: the learned policy is promising, but not universally dominant."
        ),
        "The discussion therefore separates validation from ranking": (
            "The discussion therefore separates strong ranking evidence from unsupported dominance claims. The latest same-run BER-DRL comparison includes static, round-robin, max-throughput, proportional-fair, WMMSE, queue-aware, baseline DRL, and BER-DRL under the same Rayleigh, Rician, and UMi evaluation family. BER-DRL is consistently competitive, but the max-throughput baseline remains the best observed method in Rician and UMi. This wording keeps learned-policy claims aligned with the actual evidence."
        ),
        "The third limitation is comparator completeness": (
            "The third limitation is comparator breadth rather than complete absence. The latest same-run BER-DRL comparison resolves the earlier single-method limitation for Rayleigh, Rician, and UMi, but broader traffic models, larger factory geometries, and additional random seeds are still required before claiming deployment-level policy dominance."
        ),
        "Future work should first run full same-run BER-DRL comparisons": (
            "Future work should extend the same-run BER-DRL comparison beyond the current Rayleigh, Rician, and UMi evidence by increasing random seeds, adding richer traffic arrivals, testing larger factory geometries, and evaluating hardware-aware latency. This would convert the current simulation ranking into a stronger deployment-oriented claim."
        ),
    }

    for p in body.iter(qn("w", "p")):
        text = paragraph_text(p)
        if text in replacements:
            set_paragraph_text(p, replacements[text])
        else:
            for needle, replacement in fuzzy_replacements.items():
                if text.startswith(needle):
                    set_paragraph_text(p, replacement)
                    break


def latest_ber_drl_insert(rels_root: ET.Element, content_root: ET.Element) -> list[ET.Element]:
    rows = [
        [
            "Channel",
            "Best baseline by BER",
            "BER-DRL rank",
            "BER-DRL mean BER",
            "Interpretation",
        ],
        [
            "Rayleigh",
            "queue-aware / DRL / PF / round-robin / max-throughput / WMMSE at 0 BER",
            "4",
            "0",
            "Statistically tied with the zero-BER reliability group.",
        ],
        [
            "Rician",
            "max-throughput at 0 BER",
            "2",
            "1.82e-5",
            "Strong learned policy, but not better than the best baseline.",
        ],
        [
            "TR 38.901 UMi",
            "max-throughput at 0 BER",
            "2",
            "2.60e-5",
            "Competitive with queue-aware scheduling; residual multipath errors remain.",
        ],
    ]
    elems = [
        base.make_p(
            "Latest same-run evidence update.",
            "Heading3",
            jc="both",
            line="320",
            after="80",
            bold=True,
        ),
        base.make_p(
            "The 2026-05-23 benchmark is the current thesis evidence for BER-DRL because it evaluates the trained BER-oriented policy in the same result family as the deterministic and learned baselines across Rayleigh, Rician, and TR 38.901 UMi channels. BER upper confidence remains the secondary reliability key, followed by throughput, latency, runtime, and power for engineering interpretation.",
            jc="both",
            line="320",
            after="120",
        ),
        base.make_p(
            "Table 4.3: Latest same-run BER-DRL resource-manager ranking from the 2026-05-23 Factory6G run.",
            "Caption",
            jc="center",
            line="240",
            after="80",
            bold=True,
            size="20",
        ),
        v7.make_table(rows, [1450, 2700, 1000, 1200, 3010], header=True),
    ]

    return elems


def patch_styles(tmp_dir: Path) -> None:
    styles_path = tmp_dir / "word" / "styles.xml"
    tree = ET.parse(styles_path)
    root = tree.getroot()

    def ensure_run_style(style_id: str, name: str, size: str, color: str, *, bold: bool = False) -> None:
        style = base.ensure_style(root, style_id, name)
        rpr = style.find(qn("w", "rPr"))
        if rpr is None:
            rpr = base.w_el("rPr")
            style.append(rpr)
        for child_name in ["rFonts", "sz", "szCs", "color", "b", "bCs"]:
            for old in rpr.findall(qn("w", child_name)):
                rpr.remove(old)
        base.sub(
            rpr,
            "rFonts",
            {
                qn("w", "ascii"): "Calibri",
                qn("w", "hAnsi"): "Calibri",
                qn("w", "cs"): "Calibri",
            },
        )
        base.sub(rpr, "sz", {qn("w", "val"): size})
        base.sub(rpr, "szCs", {qn("w", "val"): size})
        base.sub(rpr, "color", {qn("w", "val"): color})
        if bold:
            base.sub(rpr, "b")
            base.sub(rpr, "bCs")

    ensure_run_style("Normal", "Normal", "22", "000000")
    ensure_run_style("Heading1", "heading 1", "32", "2E74B5", bold=True)
    ensure_run_style("Heading2", "heading 2", "26", "2E74B5", bold=True)
    ensure_run_style("Heading3", "heading 3", "24", "1F4D78", bold=True)
    ensure_run_style("Caption", "caption", "20", "1F4D78")

    tree.write(styles_path, encoding="utf-8", xml_declaration=True)


def verify_heading_ladder(body: ET.Element) -> list[str]:
    headings: list[str] = []
    for p in body.iter(qn("w", "p")):
        style = p_style(p)
        text = paragraph_text(p)
        if style.startswith("Heading") and text:
            headings.append(f"{style}: {text}")
    return headings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=SRC_DOCX)
    parser.add_argument("--out", type=Path, default=OUT_DOCX)
    args = parser.parse_args()

    if not args.src.exists():
        raise SystemExit(f"Missing source DOCX: {args.src}")
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True)
    with zipfile.ZipFile(args.src) as zin:
        zin.extractall(TMP_DIR)

    document_path = TMP_DIR / "word" / "document.xml"
    document_tree = ET.parse(document_path)
    document_root = document_tree.getroot()
    body = document_root.find(qn("w", "body"))
    if body is None:
        raise SystemExit("No w:body found")

    update_year_and_methodology_headings(body)
    update_ber_drl_claims(body)
    inserted = insert_after_heading(body, "4.7 DRL and BER-DRL Validation Results", latest_ber_drl_insert(ET.Element("rels"), ET.Element("content")))
    if not inserted:
        raise SystemExit("Could not find Chapter 4.7 heading for evidence insertion")
    patch_styles(TMP_DIR)

    document_tree.write(document_path, encoding="utf-8", xml_declaration=True)
    base.write_package(TMP_DIR, args.out)

    headings = verify_heading_ladder(body)
    print(f"Wrote {args.out}")
    print(f"Heading count: {len(headings)}")
    for heading in headings[:120]:
        print(heading)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

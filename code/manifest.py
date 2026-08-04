"""Maps each paper's "Paper Name" (as used in dataset.csv) to its source
JSON file in data/papers/, so evidence text can be retrieved for every
summary sentence in the benchmark.
"""
import json
import re
import unicodedata
from pathlib import Path

# The 23 computer-vision paper JSONs store a numeric placeholder (e.g. "146")
# in their "paper_title" field instead of the real title -- titles supplied
# explicitly here, exactly as used when the dataset was built.
CV_PAPER_TITLES = {
    "91": "3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting",
    "92": "FruitNinja: 3D Object Interior Texture Generation with Gaussian Splatting",
    "93": "SinGS: Animatable Single-Image Human Gaussian Splats with Kinematic Priors",
    "94": "Template Free Reconstruction of Human-object Interaction with Procedural Interaction Generation",
    "95": "Gaussian Head Avatar: Ultra High-fidelity Head Avatar via Dynamic Gaussians",
    "96": "FATE: Full-head Gaussian Avatar with Textural Editing from Monocular Video",
    "97": "HRAvatar: High-Quality and Relightable Gaussian Head Avatar",
    "98": "HumanMM: Global Human Motion Recovery from Multi-shot Videos",
    "99": "CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models",
    "100": "Generative Inbetweening through Frame-wise Conditions-Driven Video Generation",
    "135": "CRISP: Object Pose and Shape Estimation with Test-Time Adaptation",
    "136": "ODE: Open-Set Evaluation of Hallucinations in Multimodal Large Language Models",
    "137": "VidSeg: Training-free Video Semantic Segmentation based on Diffusion Models",
    "138": "PlanarSplatting: Accurate Planar Surface Reconstruction in 3 Minutes",
    "139": "Overlooked Factors in Concept-based Explanations: Dataset Choice, Concept Learnability, and Human Capability",
    "140": "LEGO-Net: Learning Regular Rearrangements of Objects in Rooms",
    "141": "OpenScene: 3D Scene Understanding with Open Vocabularies",
    "142": "Self-Supervised Representation Learning for CAD",
    "143": "Towards Consistent Multi-Task Learning: Unlocking the Potential of Task-Specific Parameters",
    "144": "Revisiting Fairness in Multitask Learning: A Performance-Driven Approach for Variance Reduction",
    "145": "Test-Time Visual In-Context Tuning",
    "146": "SketchAgent: Language-Driven Sequential Sketch Generation",
    # 101 is a valid, complete NLP paper (ACL 2024) that happens to share a
    # numeric ID range with the CV papers -- misplaced, not miscategorized.
    "101": "Can Language Models Serve as Text-Based World Simulators?",
}

# A handful of the NLP paper JSONs have a corrupted/truncated "paper_title"
# field (an artifact of the original PDF-to-JSON parsing). Override with the
# correct title used in dataset.csv.
NLP_TITLE_OVERRIDES = {
    "1": "Transcribing Vocal Communications of Domestic Shiba Inu Dogs",
    "3": "MINPROMPT: Graph-based Minimal Prompt Data Augmentation for Few-shot Question Answering",
    "4": "SportsMetrics: Blending Text and Numerical Data to Understand Information Fusion in LLMs",
    "8": "Synchronizing Approach in Designing Annotation Guidelines for Multilingual Datasets: A COVID-19 Case Study Using English and Japanese Tweets",
    "25": "Enhancing Automated Interpretability with Output-Centric Feature Descriptions",
    "26": "SINCon: Mitigate LLM-Generated Malicious Message Injection Attack for Rumor Detection",
    "27": "Logic-Regularized Verifier Elicits Reasoning from LLMs",
    "28": "Squeezed Attention: Accelerating Long Context Length LLM Inference",
    "29": "Scalable Vision Language Model Training via High Quality Data Curation",
    "30": "Just Go Parallel: Improving the Multilingual Capabilities of Large Language Models",
    "31": "Design Choices for Extending the Context Length of Visual Language Models",
}


def normalize_title(s):
    """Normalizes a title for robust matching (case, whitespace, curly quotes)."""
    s = unicodedata.normalize("NFKD", s)
    s = s.replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"')
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_manifest(papers_dir):
    """Returns {normalized_title: (display_title, json_path)} for every
    paper JSON in `papers_dir`."""
    manifest = {}
    papers_dir = Path(papers_dir)
    for json_path in sorted(papers_dir.glob("*.json")):
        paper_id = json_path.stem
        if paper_id in CV_PAPER_TITLES:
            title = CV_PAPER_TITLES[paper_id]
        else:
            data = json.loads(json_path.read_text())
            title = NLP_TITLE_OVERRIDES.get(paper_id, data["paper_title"])
        manifest[normalize_title(title)] = (title, json_path)
    return manifest

"""Shared constants for the hallucination-detection experiments."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # .../aspect-based-hallucination-detection

# Gold-labeled sentences to classify (Sentence, Paper Name, Aspect,
# Hallucination, Temperature, ...). Contains two generations of the same 50
# papers -- Temperature="0.8" (sampled decoding) and Temperature="0"
# (greedy decoding) -- distinguished by the Temperature column; see
# dataset_utils.load_dataset_rows(temperature=...).
DATASET_CSV = ROOT / "data" / "dataset.csv"

# Source paper JSONs, used to retrieve evidence text for each summary sentence.
PAPERS_DIR = ROOT / "data" / "papers"

RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

# The 25 papers designated as the final held-out test set (see README.md and
# the paper's Dataset section). All seven models are evaluated on exactly
# this subset (478 sentences, 80 hallucinated). The other 25 papers in the
# 50-paper benchmark are not used for testing; they exist so the learned
# aggregator ("Our Model") can optionally be retrained directly on them
# (see run_our_model.py --retrain).
TEST_PAPERS = {
    "3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting",
    "FruitNinja: 3D Object Interior Texture Generation with Gaussian Splatting",
    "SinGS: Animatable Single-Image Human Gaussian Splats with Kinematic Priors",
    "Template Free Reconstruction of Human-object Interaction with Procedural Interaction Generation",
    "Gaussian Head Avatar: Ultra High-fidelity Head Avatar via Dynamic Gaussians",
    "FATE: Full-head Gaussian Avatar with Textural Editing from Monocular Video",
    "HRAvatar: High-Quality and Relightable Gaussian Head Avatar",
    "HumanMM: Global Human Motion Recovery from Multi-shot Videos",
    "CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models",
    "Generative Inbetweening through Frame-wise Conditions-Driven Video Generation",
    "How Useful is Context, Actually? Comparing LLMs and Humans on Discourse Marker Prediction",
    "ODE: Open-Set Evaluation of Hallucinations in Multimodal Large Language Models",
    "VidSeg: Training-free Video Semantic Segmentation based on Diffusion Models",
    "PlanarSplatting: Accurate Planar Surface Reconstruction in 3 Minutes",
    "What does Kiki look like? Cross-modal associations between speech sounds and visual shapes in vision-and-language models",
    "LEGO-Net: Learning Regular Rearrangements of Objects in Rooms",
    "OpenScene: 3D Scene Understanding with Open Vocabularies",
    "Self-Supervised Representation Learning for CAD",
    "Towards Consistent Multi-Task Learning: Unlocking the Potential of Task-Specific Parameters",
    "Revisiting Fairness in Multitask Learning: A Performance-Driven Approach for Variance Reduction",
    "Test-Time Visual In-Context Tuning",
    "SketchAgent: Language-Driven Sequential Sketch Generation",
    "Transcribing Vocal Communications of Domestic Shiba Inu Dogs",
    "SkillQG: Learning to Generate Question for Reading Comprehension Assessment",
    "MINPROMPT: Graph-based Minimal Prompt Data Augmentation for Few-shot Question Answering",
}

# The 25-paper held-out test set for the temp=0.8 dataset (data/
# dataset_temp08_full50.csv). See config.test_papers_for().
TEST_PAPERS_TEMP08 = {
    "3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting",
    "CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models",
    "FATE: Full-head Gaussian Avatar with Textural Editing from Monocular Video",
    "FruitNinja: 3D Object Interior Texture Generation with Gaussian Splatting",
    "Gaussian Head Avatar: Ultra High-fidelity Head Avatar via Dynamic Gaussians",
    "Generative Inbetweening through Frame-wise Conditions-Driven Video Generation",
    "HRAvatar: High-Quality and Relightable Gaussian Head Avatar",
    "How Useful is Context, Actually? Comparing LLMs and Humans on Discourse Marker Prediction",
    "HumanMM: Global Human Motion Recovery from Multi-shot Videos",
    "LEGO-Net: Learning Regular Rearrangements of Objects in Rooms",
    "MINPROMPT: Graph-based Minimal Prompt Data Augmentation for Few-shot Question Answering",
    "ODE: Open-Set Evaluation of Hallucinations in Multimodal Large Language Models",
    "OpenScene: 3D Scene Understanding with Open Vocabularies",
    "Revisiting Fairness in Multitask Learning: A Performance-Driven Approach for Variance Reduction",
    "Self-Supervised Representation Learning for CAD",
    "SinGS: Animatable Single-Image Human Gaussian Splats with Kinematic Priors",
    "SketchAgent: Language-Driven Sequential Sketch Generation",
    "SkillQG: Learning to Generate Question for Reading Comprehension Assessment",
    "Squeezed Attention: Accelerating Long Context Length LLM Inference",
    "Template Free Reconstruction of Human-object Interaction with Procedural Interaction Generation",
    "Test-Time Visual In-Context Tuning",
    "Towards Consistent Multi-Task Learning: Unlocking the Potential of Task-Specific Parameters",
    "Transcribing Vocal Communications of Domestic Shiba Inu Dogs",
    "VidSeg: Training-free Video Semantic Segmentation based on Diffusion Models",
    "What does Kiki look like? Cross-modal associations between speech sounds and visual shapes in vision-and-language models",
}

def test_papers_for(temperature):
    """Returns TEST_PAPERS_TEMP08 for temperature="0.8", else TEST_PAPERS."""
    return TEST_PAPERS_TEMP08 if str(temperature) == "0.8" else TEST_PAPERS


# GPT-4-turbo / GPT-4o baselines.
MODELS = ["gpt-4-turbo", "gpt-4o"]

# Evidence text is truncated beyond this many characters (~6k tokens) to keep
# prompts (and cost) bounded. Only used when USE_ASPECT_MAPPING is enabled,
# or by SummaC/MiniCheck, which always use aspect-restricted evidence (see
# evidence.py and README.md).
MAX_EVIDENCE_CHARS = 24000

# GPT-4/GPT-4o/Gemma/Llama/"Our Model" are given the (nearly) full paper text
# rather than the aspect-restricted mapping below, per the paper's
# Experiments section. Full papers here run ~30-38k chars, so this is a
# generous safety cap, not a routine truncation.
MAX_FULL_PAPER_CHARS = 60000

# --- Aspect-to-section evidence mapping ---
# evidence.get_evidence() implements a fixed aspect -> section mapping
# (Motivation/Paper-Goal -> Abstract+Introduction, Contribution -> +
# Conclusion, Main Results -> +Results/Experiments, Methodology -> every
# section after Related Work). The final experiments use the *full* paper
# text for GPT-4-turbo, GPT-4o, Gemma-2-2b-it, Llama-3.2-1B, and "Our Model"'s
# retrieval step, so this mapping is OPTIONAL and disabled by default
# (USE_ASPECT_MAPPING = False) for those models. Two baselines, SummaC and
# MiniCheck, deliberately opt into aspect-restricted evidence regardless of
# this flag, because their NLI/fact-checking backends are intractably slow
# against a full paper (see run_summac.py / run_minicheck.py docstrings) --
# this matches how their reported results were actually produced.
USE_ASPECT_MAPPING = False

# --- "Our Model" (local NLI-based detector) ---
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
NLI_MODEL = "microsoft/deberta-large-mnli"
TOP_N = 3
EVIDENCE_WINDOW_SIZE = 3
MIN_SIMILARITY = 0.5
MIN_CONTRADICTIONS = 2
MIN_CONTRADICTION_CONFIDENCE = 0.95
NEUTRAL_CONFIDENCE_THRESHOLD = 0.70

# Learned-aggregator hyperparameter search grid (used only by
# run_our_model.py --retrain).
C_GRID = [0.03, 0.1, 0.3, 1.0, 3.0]
THRESHOLD_GRID = [round(0.1 + 0.02 * i, 2) for i in range(41)]  # 0.10 ... 0.90

"""Generates the 10 synthetic scientific passages + 150 sentences used to
augment the learned aggregator's training data (see models/aggregator_meta.json
and README.md).

These passages are entirely fabricated (not real papers) and are NEVER part
of the released 900-sentence benchmark or any evaluation -- they exist only
to give the logistic-regression aggregator additional, unambiguous labeled
examples of entailment/contradiction/neutral relationships during training.

For each of the 10 fabricated papers, there are 15 sentences: 5 entailed by
the passage ("true", label=0), 5 that directly contradict a specific fact in
the passage (label=1), and 5 neutral -- topically related but neither
confirmed nor denied by the passage (25 of the 50 neutral sentences across
all 10 papers are labeled hallucinated, representing a plausible-sounding
but unverifiable fabrication; the other 25 are labeled not-hallucinated,
representing a true-but-unstated elaboration).

Usage:
    python generate_synthetic_data.py
"""
import csv
import json
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "synthetic"

PAPERS = [
    {
        "title": "SparseRoute: Dynamic Expert Routing for Mixture-of-Experts Transformers",
        "sections": {
            "Abstract": "We present SparseRoute, a mixture-of-experts transformer that routes each token to exactly 2 of 32 experts using a learned load-balancing gate. SparseRoute reduces training FLOPs by 41% relative to a dense transformer of equal parameter count while matching its perplexity on the WikiText-103 benchmark.",
            "Introduction": "Dense transformers activate every parameter for every token, wasting computation on tokens that only need a small subset of the model's capacity. Prior mixture-of-experts approaches suffer from expert collapse, where a small number of experts receive most of the routed tokens.",
            "Methodology": "SparseRoute routes each token to its top-2 experts out of 32 total experts, selected by a lightweight linear gate trained jointly with the experts. An auxiliary load-balancing loss with coefficient 0.01 penalizes uneven expert usage across a training batch.",
            "Main Results": "On WikiText-103, SparseRoute reaches a validation perplexity of 18.4, within 0.2 points of a dense transformer baseline with the same parameter count, while using 41% fewer training FLOPs. Ablating the load-balancing loss increases the FLOP reduction to 55% but raises perplexity to 21.7.",
            "Conclusion": "SparseRoute demonstrates that top-2 expert routing with load balancing can match dense-model quality at substantially lower training cost.",
        },
    },
    {
        "title": "MolGraphNet: Graph Neural Networks for Molecular Property Prediction",
        "sections": {
            "Abstract": "MolGraphNet is a graph neural network that predicts molecular properties directly from atomic graphs, using 5 message-passing layers with edge-conditioned convolutions. On the QM9 benchmark, MolGraphNet achieves a mean absolute error of 0.038 eV for HOMO-LUMO gap prediction, improving over the previous best graph-based method by 12%.",
            "Introduction": "Predicting quantum-chemical properties from molecular structure typically requires expensive density functional theory calculations. Graph neural networks offer a faster surrogate, but existing methods struggle to capture long-range interactions between distant atoms in large molecules.",
            "Methodology": "MolGraphNet represents each molecule as a graph with atoms as nodes and bonds as edges, applying 5 layers of edge-conditioned message passing followed by a set2set readout to produce a fixed-size molecular embedding. Training uses the Adam optimizer with a learning rate of 5e-4 for 300 epochs.",
            "Main Results": "MolGraphNet attains a mean absolute error of 0.038 eV on HOMO-LUMO gap prediction on QM9, a 12% improvement over the strongest prior graph-based baseline. On dipole moment prediction, MolGraphNet's error is 0.021 Debye, roughly matching the previous state of the art.",
            "Conclusion": "Edge-conditioned message passing with 5 layers is sufficient to substantially improve HOMO-LUMO gap prediction without architectural changes to capture longer-range interactions.",
        },
    },
    {
        "title": "EchoDiff: Diffusion Models for Text-to-Audio Synthesis",
        "sections": {
            "Abstract": "EchoDiff generates audio waveforms conditioned on text descriptions using a latent diffusion model with 4 denoising steps at inference time. Human evaluators preferred EchoDiff's outputs over the AudioLDM baseline in 64% of pairwise comparisons.",
            "Introduction": "Text-to-audio generation must model both the acoustic content described by the text and realistic waveform-level detail. Prior diffusion-based approaches require 50 or more denoising steps at inference, making generation slow.",
            "Methodology": "EchoDiff performs diffusion in a compressed latent space produced by a pretrained audio autoencoder, and distills a 50-step teacher diffusion process down to a 4-step student using progressive distillation. Text conditioning is injected via cross-attention at every denoising step.",
            "Main Results": "In a listening study with 40 participants, EchoDiff's 4-step outputs were preferred over AudioLDM's 50-step outputs in 64% of pairwise comparisons, while generating audio 11 times faster. Objective Frechet Audio Distance was 2.1 for EchoDiff versus 2.6 for AudioLDM.",
            "Conclusion": "Progressive distillation to 4 denoising steps preserves and even improves perceived audio quality while greatly reducing inference cost.",
        },
    },
    {
        "title": "FewShotDet: Few-Shot Object Detection via Prototype Alignment",
        "sections": {
            "Abstract": "FewShotDet adapts an object detector to novel categories from just 5 labeled examples per class by aligning region features to class prototypes computed from the support set. On the PASCAL VOC few-shot split, FewShotDet improves novel-class average precision by 7.3 points over the previous best method.",
            "Introduction": "Standard object detectors require thousands of labeled examples per class, but many real-world categories only have a handful of available examples. Existing few-shot detectors often overfit to the small support set and generalize poorly to query images with different viewpoints.",
            "Methodology": "FewShotDet computes a prototype embedding for each novel class by averaging region-of-interest features from 5 support examples, then aligns query region features to these prototypes using a learned metric with margin-based contrastive loss. The base detector backbone is frozen during few-shot adaptation.",
            "Main Results": "On the PASCAL VOC few-shot benchmark with 5 examples per novel class, FewShotDet achieves 46.2 average precision on novel classes, a 7.3-point improvement over the strongest prior baseline. Base-class performance drops by only 0.4 points relative to the fully-supervised detector.",
            "Conclusion": "Freezing the backbone and aligning region features to class prototypes prevents overfitting to the small support set while still adapting effectively to novel categories.",
        },
    },
    {
        "title": "ContraMed: Contrastive Pretraining for Medical Image Classification",
        "sections": {
            "Abstract": "ContraMed pretrains a vision encoder using contrastive learning on 2.1 million unlabeled chest X-ray images, then fine-tunes on a small labeled pneumonia classification dataset. ContraMed reaches 91.4% accuracy using only 500 labeled images, matching a fully-supervised baseline trained on 10,000 labeled images.",
            "Introduction": "Labeled medical images are scarce and expensive to annotate, since annotation requires expert radiologists. Self-supervised pretraining on unlabeled images is a promising way to reduce the amount of labeled data needed for downstream diagnosis tasks.",
            "Methodology": "ContraMed pretrains a ResNet-50 encoder with a contrastive loss that pulls together two augmented views of the same X-ray while pushing apart views from different images, using a batch size of 1024 for 200 pretraining epochs. Fine-tuning attaches a linear classification head and updates the full network on the labeled pneumonia dataset.",
            "Main Results": "With only 500 labeled images, ContraMed's fine-tuned classifier reaches 91.4% accuracy on the held-out pneumonia test set, matching a fully-supervised ResNet-50 baseline trained on 10,000 labeled images (91.7% accuracy). Without contrastive pretraining, the same architecture trained on 500 labeled images only reaches 78.2% accuracy.",
            "Conclusion": "Large-scale contrastive pretraining on unlabeled chest X-rays can reduce the labeled data requirement for pneumonia classification by roughly 20-fold.",
        },
    },
    {
        "title": "QuantEdge: Post-Training Quantization for Edge-Deployed Vision Models",
        "sections": {
            "Abstract": "QuantEdge is a post-training quantization method that compresses vision model weights and activations to 4 bits without any retraining, using per-channel calibration on 512 unlabeled images. QuantEdge preserves 99.1% of the original ImageNet top-1 accuracy while reducing model size by 7.2 times.",
            "Introduction": "Deploying vision models on edge devices requires reducing both memory footprint and inference latency, but quantization to very low bit-widths typically causes substantial accuracy degradation, especially without expensive retraining.",
            "Methodology": "QuantEdge calibrates per-channel quantization scales for both weights and activations using 512 unlabeled calibration images, then applies a bias-correction step that adjusts each layer's output statistics to match the full-precision model. No gradient-based retraining or fine-tuning is used at any stage.",
            "Main Results": "QuantEdge quantizes ResNet-50 to 4 bits while retaining 99.1% of the original top-1 ImageNet accuracy (75.6% quantized versus 76.3% full-precision), a 7.2x reduction in model size. Without the bias-correction step, accuracy drops to 71.8%.",
            "Conclusion": "Per-channel calibration combined with bias correction allows 4-bit post-training quantization without any retraining, at a small accuracy cost.",
        },
    },
    {
        "title": "RetroQA: Retrieval-Augmented Question Answering with Dynamic Context Windows",
        "sections": {
            "Abstract": "RetroQA answers open-domain questions by dynamically adjusting the number of retrieved passages per question, using a lightweight relevance classifier to decide when enough evidence has been gathered. On Natural Questions, RetroQA improves exact-match accuracy by 4.6 points over a fixed-top-10-passage baseline while retrieving 35% fewer passages on average.",
            "Introduction": "Retrieval-augmented question answering systems typically retrieve a fixed number of passages for every question, which wastes computation on easy questions and under-retrieves for questions that require synthesizing evidence from many sources.",
            "Methodology": "RetroQA retrieves passages one at a time, and after each retrieval step a lightweight classifier predicts whether the accumulated evidence is sufficient to answer the question; retrieval stops once the classifier's confidence exceeds 0.85. The relevance classifier is trained on 15,000 question-evidence-sufficiency labels.",
            "Main Results": "RetroQA improves exact-match accuracy on Natural Questions from 41.2% (fixed top-10 baseline) to 45.8%, while retrieving an average of 6.5 passages per question instead of a fixed 10, a 35% reduction. On questions requiring multi-hop reasoning, the improvement grows to 8.1 points.",
            "Conclusion": "Deciding retrieval depth dynamically per question improves both accuracy and retrieval efficiency compared to a fixed retrieval budget.",
        },
    },
    {
        "title": "NeuroPDE: Neural Operators for Solving Parametric PDEs",
        "sections": {
            "Abstract": "NeuroPDE learns a neural operator that maps PDE initial conditions directly to solutions across a family of parametric partial differential equations, trained on 10,000 simulated trajectories of the 2D Navier-Stokes equations. NeuroPDE solves new instances 800 times faster than a classical numerical solver while keeping relative error below 2%.",
            "Introduction": "Classical numerical solvers for parametric PDEs must be re-run from scratch for every new set of initial conditions or parameters, which is prohibitively slow for applications requiring many repeated solves, such as design optimization.",
            "Methodology": "NeuroPDE uses a Fourier neural operator architecture with 4 spectral convolution layers to learn a mapping from initial vorticity fields to solution trajectories at future timesteps, trained on 10,000 simulated 2D Navier-Stokes trajectories at a fixed viscosity of 1e-3. Training minimizes relative L2 error between predicted and ground-truth trajectories.",
            "Main Results": "On held-out initial conditions, NeuroPDE achieves a relative L2 error of 1.8% and produces a full solution trajectory in 45 milliseconds, compared to 36 seconds for a classical spectral solver, an 800x speedup. Error increases to 6.4% when evaluated at a viscosity value not seen during training.",
            "Conclusion": "Fourier neural operators trained on a fixed viscosity generalize well within that regime but require retraining or fine-tuning to generalize across viscosity values.",
        },
    },
    {
        "title": "EmoSpeech: Multimodal Speech Emotion Recognition with Cross-Attention",
        "sections": {
            "Abstract": "EmoSpeech recognizes emotion from speech by jointly modeling acoustic features and the transcribed text using cross-attention between the two modalities. On the IEMOCAP benchmark, EmoSpeech achieves 74.9% weighted accuracy across 4 emotion categories, outperforming an audio-only baseline by 6.1 points.",
            "Introduction": "Speech emotion recognition systems that rely solely on acoustic features miss lexical cues present in what is actually said, while text-only systems miss prosodic cues like pitch and intonation.",
            "Methodology": "EmoSpeech encodes audio with a pretrained wav2vec 2.0 model and text with a pretrained BERT encoder, then fuses the two modalities using 2 layers of cross-attention where each modality attends to the other before a final classification layer over 4 emotion categories (angry, happy, sad, neutral).",
            "Main Results": "EmoSpeech reaches 74.9% weighted accuracy on IEMOCAP's 4-class emotion recognition task, compared to 68.8% for an audio-only baseline and 65.3% for a text-only baseline. The angry category shows the largest gain from multimodal fusion, improving by 9.4 points over the audio-only baseline.",
            "Conclusion": "Cross-attention fusion of acoustic and lexical information improves emotion recognition accuracy over either modality alone, with the largest gains on emotions that have strong lexical cues.",
        },
    },
    {
        "title": "FedCompress: Communication-Efficient Federated Learning via Gradient Compression",
        "sections": {
            "Abstract": "FedCompress reduces the communication cost of federated learning by compressing client gradient updates to 8 bits per parameter using error-feedback quantization before transmission to the server. Across 100 simulated clients on the CIFAR-10 federated benchmark, FedCompress reduces total communication by 3.8 times while matching the accuracy of uncompressed federated averaging.",
            "Introduction": "Federated learning requires many rounds of communication between clients and a central server, and this communication cost, not local computation, is usually the primary bottleneck, especially for clients on limited bandwidth connections.",
            "Methodology": "FedCompress quantizes each client's gradient update to 8 bits per parameter before transmission, and accumulates the quantization error locally to add it back into the next round's update, preventing compression error from accumulating unboundedly across rounds. The server aggregates compressed updates using standard federated averaging.",
            "Main Results": "On the CIFAR-10 federated benchmark with 100 clients, FedCompress reaches 82.1% test accuracy after 200 communication rounds, within 0.3 points of uncompressed federated averaging (82.4%), while transmitting 3.8 times fewer bits in total. Without error feedback, accuracy drops to 76.5% due to accumulated quantization error.",
            "Conclusion": "Error-feedback quantization allows aggressive 8-bit gradient compression in federated learning with almost no accuracy cost, provided the accumulated quantization error is fed back into subsequent rounds.",
        },
    },
]

# 15 sentences per paper: (aspect, sentence, label, kind).
# kind in {"true", "contradict", "neutral"} -- for bookkeeping only, not
# consumed by any training code; label is what's actually used.
SENTENCES = [
    # 1. SparseRoute
    ("Motivation", "Dense transformers activate all their parameters for every token, which wastes computation on tokens that only need a fraction of the model's capacity.", 0, "true"),
    ("Contribution", "SparseRoute is a mixture-of-experts transformer that routes each token to its top-2 experts out of a pool of 32.", 0, "true"),
    ("Methodology", "An auxiliary load-balancing loss with a coefficient of 0.01 discourages uneven expert usage across a batch.", 0, "true"),
    ("Main Results", "SparseRoute reduces training FLOPs by 41% while reaching a validation perplexity within 0.2 points of a dense transformer baseline.", 0, "true"),
    ("Paper Goal", "The goal of this work is to match dense-transformer quality at lower training cost via sparse expert routing.", 0, "true"),
    ("Motivation", "Prior mixture-of-experts methods are praised in this paper for evenly distributing tokens across experts without any additional loss terms.", 1, "contradict"),
    ("Contribution", "SparseRoute routes each token to all 32 experts simultaneously to maximize representational capacity.", 1, "contradict"),
    ("Methodology", "The load-balancing loss coefficient is set to 10.0, an order of magnitude larger than the expert-routing loss itself.", 1, "contradict"),
    ("Main Results", "SparseRoute increases training FLOPs by 41% relative to a dense transformer while achieving substantially worse perplexity.", 1, "contradict"),
    ("Paper Goal", "This paper's main goal is to show that mixture-of-experts models are strictly worse than dense transformers in every setting.", 1, "contradict"),
    ("Methodology", "SparseRoute's gating mechanism could plausibly be extended to route based on both token identity and sequence position.", 0, "neutral"),
    ("Main Results", "The 41% FLOP reduction reported for SparseRoute was measured using a batch size of 256 sequences.", 1, "neutral"),
    ("Contribution", "SparseRoute's routing gate is implemented as a single linear layer followed by a softmax over the 32 experts.", 0, "neutral"),
    ("Main Results", "SparseRoute's authors also report a 3.2x inference speedup on a single A100 GPU compared to the dense baseline.", 1, "neutral"),
    ("Motivation", "Expert collapse in mixture-of-experts models has previously been linked to poor initialization of the gating network's weights.", 1, "neutral"),

    # 2. MolGraphNet
    ("Motivation", "Computing quantum-chemical molecular properties with density functional theory is computationally expensive, motivating faster graph-based surrogates.", 0, "true"),
    ("Contribution", "MolGraphNet uses 5 layers of edge-conditioned message passing to predict molecular properties from atomic graphs.", 0, "true"),
    ("Methodology", "MolGraphNet is trained with the Adam optimizer at a learning rate of 5e-4 for 300 epochs.", 0, "true"),
    ("Main Results", "MolGraphNet achieves a mean absolute error of 0.038 eV on HOMO-LUMO gap prediction, a 12% improvement over the best prior graph-based method.", 0, "true"),
    ("Paper Goal", "This paper aims to improve graph neural network accuracy on quantum-chemical property prediction, specifically the HOMO-LUMO gap.", 0, "true"),
    ("Contribution", "MolGraphNet relies on 20 layers of message passing to achieve its reported accuracy.", 1, "contradict"),
    ("Methodology", "MolGraphNet is trained without any gradient-based optimizer, instead using a closed-form least-squares solution.", 1, "contradict"),
    ("Main Results", "MolGraphNet's HOMO-LUMO gap prediction error is 12% worse than the previous best graph-based method.", 1, "contradict"),
    ("Main Results", "On dipole moment prediction, MolGraphNet's error is dramatically higher than all prior methods.", 1, "contradict"),
    ("Motivation", "The paper argues that density functional theory calculations are faster than any graph neural network surrogate could ever be.", 1, "contradict"),
    ("Methodology", "The set2set readout function used in MolGraphNet could likely be replaced with a simple sum-pooling operation with modest accuracy loss.", 0, "neutral"),
    ("Main Results", "MolGraphNet was also evaluated on the Tox21 toxicity prediction benchmark, achieving competitive results.", 1, "neutral"),
    ("Contribution", "MolGraphNet's edge-conditioned convolutions incorporate bond-type information as part of the edge features.", 0, "neutral"),
    ("Main Results", "MolGraphNet's authors report that training took approximately 14 hours on a single GPU.", 1, "neutral"),
    ("Motivation", "Long-range interactions between distant atoms are especially important for predicting properties of large, flexible molecules.", 1, "neutral"),

    # 3. EchoDiff
    ("Motivation", "Prior diffusion-based text-to-audio methods require 50 or more denoising steps, making generation slow.", 0, "true"),
    ("Contribution", "EchoDiff distills a 50-step diffusion process down to a 4-step student model using progressive distillation.", 0, "true"),
    ("Methodology", "Text conditioning in EchoDiff is injected via cross-attention at every denoising step.", 0, "true"),
    ("Main Results", "In a listening study, EchoDiff's outputs were preferred over AudioLDM's in 64% of pairwise comparisons, while being 11 times faster.", 0, "true"),
    ("Paper Goal", "This work aims to substantially speed up diffusion-based text-to-audio generation without sacrificing perceived quality.", 0, "true"),
    ("Contribution", "EchoDiff requires 100 denoising steps at inference time, more than twice as many as the AudioLDM baseline.", 1, "contradict"),
    ("Main Results", "Human evaluators preferred AudioLDM's outputs over EchoDiff's in 64% of comparisons.", 1, "contradict"),
    ("Main Results", "EchoDiff's Frechet Audio Distance of 2.1 is worse than AudioLDM's score of 2.6, since lower FAD is better.", 1, "contradict"),
    ("Methodology", "EchoDiff performs diffusion directly on raw waveform samples rather than in any compressed latent space.", 1, "contradict"),
    ("Motivation", "The paper claims that inference speed is irrelevant to the practical usefulness of text-to-audio systems.", 1, "contradict"),
    ("Methodology", "The audio autoencoder used to produce EchoDiff's latent space was likely pretrained on a separate large audio corpus.", 0, "neutral"),
    ("Main Results", "The listening study for EchoDiff was conducted using headphones in a soundproofed room.", 1, "neutral"),
    ("Contribution", "Progressive distillation as used in EchoDiff was originally proposed for image diffusion models before being applied to audio.", 0, "neutral"),
    ("Main Results", "EchoDiff also supports generating audio longer than 30 seconds without any quality degradation.", 1, "neutral"),
    ("Motivation", "Latent-space diffusion models are generally more memory-efficient to train than raw-waveform diffusion models.", 1, "neutral"),

    # 4. FewShotDet
    ("Motivation", "Standard object detectors need thousands of labeled examples per class, which is impractical for many real-world categories.", 0, "true"),
    ("Contribution", "FewShotDet aligns region features to class prototypes computed from just 5 labeled support examples per novel class.", 0, "true"),
    ("Methodology", "The base detector backbone in FewShotDet is kept frozen during few-shot adaptation to novel classes.", 0, "true"),
    ("Main Results", "FewShotDet improves novel-class average precision by 7.3 points over the previous best method on the PASCAL VOC few-shot split.", 0, "true"),
    ("Paper Goal", "The goal of FewShotDet is to adapt an object detector to novel categories using only a handful of labeled examples per class.", 0, "true"),
    ("Contribution", "FewShotDet requires fine-tuning the entire detector backbone on 1,000 examples per novel class.", 1, "contradict"),
    ("Main Results", "FewShotDet's novel-class average precision is 7.3 points worse than the previous best method.", 1, "contradict"),
    ("Main Results", "Base-class performance drops by 15 points when FewShotDet is adapted to novel categories.", 1, "contradict"),
    ("Methodology", "FewShotDet computes class prototypes by averaging features from over 500 examples per class.", 1, "contradict"),
    ("Motivation", "The paper argues that few-shot object detection is already a solved problem with no remaining challenges.", 1, "contradict"),
    ("Methodology", "The margin-based contrastive loss used in FewShotDet likely has a hyperparameter controlling the margin size.", 0, "neutral"),
    ("Main Results", "FewShotDet was also tested on the COCO few-shot benchmark with similar relative improvements.", 1, "neutral"),
    ("Contribution", "FewShotDet's prototype alignment approach could plausibly generalize to few-shot instance segmentation as well.", 0, "neutral"),
    ("Main Results", "FewShotDet's authors report that inference takes under 50 milliseconds per image on a modern GPU.", 1, "neutral"),
    ("Motivation", "Novel categories in real-world applications often differ substantially in viewpoint and scale from typical training examples.", 1, "neutral"),

    # 5. ContraMed
    ("Motivation", "Labeled medical images are expensive to obtain because annotation requires expert radiologists.", 0, "true"),
    ("Contribution", "ContraMed pretrains a vision encoder with contrastive learning on 2.1 million unlabeled chest X-rays.", 0, "true"),
    ("Methodology", "ContraMed's contrastive pretraining uses a batch size of 1024 for 200 epochs.", 0, "true"),
    ("Main Results", "With only 500 labeled images, ContraMed matches the accuracy of a fully-supervised baseline trained on 10,000 labeled images.", 0, "true"),
    ("Paper Goal", "This paper aims to reduce the amount of labeled data needed for medical image classification using self-supervised pretraining.", 0, "true"),
    ("Contribution", "ContraMed pretrains its encoder using full supervision on 2.1 million labeled chest X-rays.", 1, "contradict"),
    ("Main Results", "ContraMed requires 10,000 labeled images to match the accuracy of a baseline trained on only 500 labeled images.", 1, "contradict"),
    ("Main Results", "Without contrastive pretraining, the same architecture trained on 500 labeled images reaches 95% accuracy, higher than with pretraining.", 1, "contradict"),
    ("Methodology", "ContraMed pretrains using a batch size of 8, since larger batches were found to destabilize contrastive training.", 1, "contradict"),
    ("Motivation", "The paper claims that expert radiologist annotation is fast and inexpensive to obtain at scale.", 1, "contradict"),
    ("Methodology", "The ResNet-50 encoder used in ContraMed could plausibly be replaced with a vision transformer backbone.", 0, "neutral"),
    ("Main Results", "ContraMed was also evaluated on a tuberculosis detection dataset with similarly strong results.", 1, "neutral"),
    ("Contribution", "ContraMed's contrastive loss pulls together two augmented views of the same X-ray image.", 0, "neutral"),
    ("Main Results", "ContraMed's pretraining phase took approximately one week on 8 GPUs.", 1, "neutral"),
    ("Motivation", "Self-supervised pretraining has also shown promise in natural image domains beyond medical imaging.", 1, "neutral"),

    # 6. QuantEdge
    ("Motivation", "Deploying vision models on edge devices requires reducing both memory footprint and inference latency.", 0, "true"),
    ("Contribution", "QuantEdge quantizes model weights and activations to 4 bits without any retraining.", 0, "true"),
    ("Methodology", "QuantEdge uses 512 unlabeled images for per-channel calibration.", 0, "true"),
    ("Main Results", "QuantEdge preserves 99.1% of the original ImageNet top-1 accuracy while reducing model size by 7.2 times.", 0, "true"),
    ("Paper Goal", "This paper aims to achieve aggressive low-bit quantization of vision models without any retraining.", 0, "true"),
    ("Contribution", "QuantEdge requires full retraining from scratch to reach its reported accuracy after quantization.", 1, "contradict"),
    ("Methodology", "QuantEdge calibrates using 500,000 labeled images rather than a small unlabeled calibration set.", 1, "contradict"),
    ("Main Results", "QuantEdge's 4-bit model retains only 60% of the original top-1 accuracy.", 1, "contradict"),
    ("Main Results", "Without bias correction, QuantEdge's accuracy actually improves relative to the full method.", 1, "contradict"),
    ("Motivation", "The paper argues that memory footprint is irrelevant for edge deployment of vision models.", 1, "contradict"),
    ("Methodology", "The bias-correction step in QuantEdge likely operates independently on each layer of the network.", 0, "neutral"),
    ("Main Results", "QuantEdge was also tested on a mobile object detection model with comparable compression ratios.", 1, "neutral"),
    ("Contribution", "QuantEdge's per-channel calibration assigns a separate quantization scale to each output channel.", 0, "neutral"),
    ("Main Results", "QuantEdge's calibration procedure takes under one minute on a single CPU core.", 1, "neutral"),
    ("Motivation", "Lower bit-widths generally reduce both memory bandwidth and energy consumption during inference.", 0, "neutral"),

    # 7. RetroQA
    ("Motivation", "Fixed-size passage retrieval wastes computation on easy questions and under-retrieves for questions needing many sources.", 0, "true"),
    ("Contribution", "RetroQA dynamically decides how many passages to retrieve per question using a lightweight relevance classifier.", 0, "true"),
    ("Methodology", "RetroQA stops retrieving once the relevance classifier's confidence exceeds 0.85.", 0, "true"),
    ("Main Results", "RetroQA improves exact-match accuracy on Natural Questions from 41.2% to 45.8% while retrieving 35% fewer passages on average.", 0, "true"),
    ("Paper Goal", "This paper aims to make retrieval depth adaptive per question rather than fixed for all questions.", 0, "true"),
    ("Contribution", "RetroQA always retrieves exactly 10 passages for every question regardless of difficulty.", 1, "contradict"),
    ("Main Results", "RetroQA's exact-match accuracy on Natural Questions drops from 45.8% down to 41.2% relative to the fixed baseline.", 1, "contradict"),
    ("Main Results", "RetroQA retrieves 35% more passages on average than the fixed top-10 baseline.", 1, "contradict"),
    ("Methodology", "The relevance classifier in RetroQA is trained on over 10 million labeled examples.", 1, "contradict"),
    ("Motivation", "The paper claims multi-hop questions require no more evidence than simple single-hop questions.", 1, "contradict"),
    ("Methodology", "The relevance classifier used by RetroQA could plausibly be a small transformer encoder.", 0, "neutral"),
    ("Main Results", "RetroQA was also evaluated on the TriviaQA benchmark with similar efficiency gains.", 1, "neutral"),
    ("Contribution", "RetroQA retrieves passages one at a time rather than all at once.", 0, "neutral"),
    ("Main Results", "RetroQA's relevance classifier adds less than 5 milliseconds of latency per retrieval step.", 1, "neutral"),
    ("Motivation", "Multi-hop reasoning questions often require synthesizing evidence spread across multiple documents.", 0, "neutral"),

    # 8. NeuroPDE
    ("Motivation", "Classical numerical PDE solvers must be re-run from scratch for every new set of initial conditions, which is slow for repeated solves.", 0, "true"),
    ("Contribution", "NeuroPDE uses a Fourier neural operator with 4 spectral convolution layers to map initial conditions to PDE solutions.", 0, "true"),
    ("Methodology", "NeuroPDE is trained on 10,000 simulated 2D Navier-Stokes trajectories at a fixed viscosity of 1e-3.", 0, "true"),
    ("Main Results", "NeuroPDE solves new instances 800 times faster than a classical solver while keeping relative error below 2%.", 0, "true"),
    ("Paper Goal", "This paper aims to accelerate solving parametric PDEs by learning a neural operator instead of using classical numerical methods.", 0, "true"),
    ("Contribution", "NeuroPDE uses 40 spectral convolution layers, ten times more than actually reported.", 1, "contradict"),
    ("Main Results", "NeuroPDE is 800 times slower than a classical spectral solver at producing new solutions.", 1, "contradict"),
    ("Main Results", "NeuroPDE's relative L2 error exceeds 50% on held-out initial conditions within the training viscosity regime.", 1, "contradict"),
    ("Methodology", "NeuroPDE was trained on real experimental fluid-dynamics measurements rather than simulated trajectories.", 1, "contradict"),
    ("Motivation", "The paper claims classical numerical solvers are always faster than any learned neural operator.", 1, "contradict"),
    ("Methodology", "The Fourier neural operator architecture likely leverages the fast Fourier transform for efficient spectral convolutions.", 0, "neutral"),
    ("Main Results", "NeuroPDE was also tested on 3D Navier-Stokes simulations with comparable speedups.", 1, "neutral"),
    ("Contribution", "NeuroPDE's training objective minimizes relative L2 error between predicted and ground-truth trajectories.", 0, "neutral"),
    ("Main Results", "NeuroPDE's training process required approximately 48 hours on 4 GPUs.", 1, "neutral"),
    ("Motivation", "Design optimization workflows often require solving the same PDE family under many different parameter settings.", 0, "neutral"),

    # 9. EmoSpeech
    ("Motivation", "Audio-only speech emotion systems miss lexical cues, while text-only systems miss prosodic cues like pitch and intonation.", 0, "true"),
    ("Contribution", "EmoSpeech fuses acoustic and text features using 2 layers of cross-attention between the two modalities.", 0, "true"),
    ("Methodology", "EmoSpeech encodes audio with a pretrained wav2vec 2.0 model and text with a pretrained BERT encoder.", 0, "true"),
    ("Main Results", "EmoSpeech reaches 74.9% weighted accuracy on IEMOCAP, outperforming an audio-only baseline by 6.1 points.", 0, "true"),
    ("Paper Goal", "This paper aims to improve speech emotion recognition by jointly modeling acoustic and lexical information.", 0, "true"),
    ("Contribution", "EmoSpeech relies exclusively on acoustic features and does not use any text encoder at all.", 1, "contradict"),
    ("Main Results", "EmoSpeech's weighted accuracy of 74.9% is lower than the audio-only baseline's 68.8%.", 1, "contradict"),
    ("Main Results", "The angry emotion category shows the smallest gain from multimodal fusion, among all 4 categories.", 1, "contradict"),
    ("Methodology", "EmoSpeech classifies emotion into 12 fine-grained categories rather than 4 broad ones.", 1, "contradict"),
    ("Motivation", "The paper argues that prosodic cues such as pitch carry no useful information for emotion recognition.", 1, "contradict"),
    ("Methodology", "The wav2vec 2.0 encoder used in EmoSpeech was likely pretrained on a large unlabeled speech corpus before fine-tuning.", 0, "neutral"),
    ("Main Results", "EmoSpeech was also evaluated on the MELD dataset with consistent relative improvements.", 1, "neutral"),
    ("Contribution", "EmoSpeech's cross-attention layers allow each modality to attend to the other before final classification.", 0, "neutral"),
    ("Main Results", "EmoSpeech's total parameter count is under 10 million, making it lightweight enough for mobile deployment.", 1, "neutral"),
    ("Motivation", "Prosodic features such as pitch and intonation are known to carry emotional information independent of word choice.", 0, "neutral"),

    # 10. FedCompress
    ("Motivation", "Communication between clients and the server, not local computation, is usually the primary bottleneck in federated learning.", 0, "true"),
    ("Contribution", "FedCompress compresses client gradient updates to 8 bits per parameter using error-feedback quantization.", 0, "true"),
    ("Methodology", "FedCompress accumulates quantization error locally and adds it back into the next round's update.", 0, "true"),
    ("Main Results", "FedCompress reduces total communication by 3.8 times while matching the accuracy of uncompressed federated averaging.", 0, "true"),
    ("Paper Goal", "This paper aims to reduce the communication cost of federated learning without sacrificing model accuracy.", 0, "true"),
    ("Contribution", "FedCompress transmits full 32-bit gradients with no compression of any kind.", 1, "contradict"),
    ("Main Results", "FedCompress's compressed model reaches only 50% test accuracy, far below uncompressed federated averaging.", 1, "contradict"),
    ("Main Results", "Without error feedback, FedCompress's accuracy actually improves relative to using error feedback.", 1, "contradict"),
    ("Methodology", "FedCompress discards accumulated quantization error at the end of every round instead of feeding it back.", 1, "contradict"),
    ("Motivation", "The paper claims that client-side computation, not communication, is the dominant bottleneck in federated learning.", 1, "contradict"),
    ("Methodology", "The 8-bit quantization scheme in FedCompress could plausibly be extended to even lower bit-widths with further error-feedback tuning.", 0, "neutral"),
    ("Main Results", "FedCompress was also evaluated on a federated speech recognition benchmark with similar communication savings.", 1, "neutral"),
    ("Contribution", "FedCompress's server aggregates compressed client updates using standard federated averaging.", 0, "neutral"),
    ("Main Results", "FedCompress's 200 communication rounds took approximately 6 hours to complete in simulation.", 1, "neutral"),
    ("Motivation", "Clients on limited-bandwidth mobile networks are especially sensitive to the total amount of data transmitted per round.", 0, "neutral"),
]


def to_paper_json(paper_def):
    sections = [{"section": name, "chunks": [text]} for name, text in paper_def["sections"].items()]
    return {"paper_title": paper_def["title"], "sections": sections}


def main():
    assert len(SENTENCES) == len(PAPERS) * 15 == 150

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    idx = 0
    for i, paper in enumerate(PAPERS, 1):
        paper_json = to_paper_json(paper)
        with open(OUT_DIR / f"synth_{i:02d}.json", "w") as f:
            json.dump(paper_json, f, indent=2)
        for _ in range(15):
            aspect, sentence, label, kind = SENTENCES[idx]
            idx += 1
            rows.append({
                "Paper Name": paper["title"], "Aspect": aspect, "Sentence": sentence,
                "Hallucination": label, "kind": kind,
            })

    out_csv = OUT_DIR / "synthetic_sentences.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Paper Name", "Aspect", "Sentence", "Hallucination", "kind"])
        writer.writeheader()
        writer.writerows(rows)

    n_true = sum(1 for _, _, _, k in SENTENCES if k == "true")
    n_contra = sum(1 for _, _, _, k in SENTENCES if k == "contradict")
    n_neutral_pos = sum(1 for _, _, l, k in SENTENCES if k == "neutral" and l == 1)
    n_neutral_neg = sum(1 for _, _, l, k in SENTENCES if k == "neutral" and l == 0)
    print(f"Wrote {len(PAPERS)} paper JSONs and {len(rows)} sentences to {OUT_DIR}")
    print(f"  true={n_true} (label=0) / contradict={n_contra} (label=1) / "
          f"neutral={n_neutral_pos + n_neutral_neg} ({n_neutral_pos} label=1, {n_neutral_neg} label=0)")


if __name__ == "__main__":
    main()

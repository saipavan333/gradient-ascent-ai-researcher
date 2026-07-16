/* ============================================================================
   Gradient Ascent — site engine
   - single source of truth for the curriculum
   - renders sidebar, breadcrumb, prev/next (always lands at top)
   - interactive quizzes, collapsible solutions, theme, progress, math
   ========================================================================== */
(function () {
  "use strict";

  /* --------------------------------------------------------------------------
     CURRICULUM  (the learning path; order matters — never depends on later)
     status: "done"  -> built & linkable
             "soon"  -> planned, shown greyed
     file paths are relative to the site root.
     -------------------------------------------------------------------------- */
  const T1 = "lessons/track-1-linear-algebra-probability/";
  const TRACKS = [
    {
      n: 1, id: "t1", title: "Linear Algebra & Probability",
      desc: "The language of ML — vectors, matrices, and uncertainty, built from geometric intuition.",
      lessons: [
        { id: "t1-l00", title: "Overview & how to learn this",          file: T1 + "00-overview.html",              status: "done" },
        { id: "t1-l01", title: "Vectors: the atoms of ML",              file: T1 + "01-vectors.html",               status: "done" },
        { id: "t1-l02", title: "Vector operations & geometry",          file: T1 + "02-vector-operations.html",     status: "done" },
        { id: "t1-l03", title: "The dot product: length, angle, projection", file: T1 + "03-dot-product.html",       status: "done" },
        { id: "t1-l04", title: "Linear combinations, span & basis",     file: T1 + "04-span-basis.html",            status: "done" },
        { id: "t1-l05", title: "Matrices as linear transformations",    file: T1 + "05-matrices-as-maps.html",      status: "done" },
        { id: "t1-l06", title: "Matrix multiplication = composition",   file: T1 + "06-matrix-multiplication.html", status: "done" },
        { id: "t1-l07", title: "Determinant: area, volume, invertibility", file: T1 + "07-determinant.html", status: "done" },
        { id: "t1-l08", title: "Eigenvectors & eigenvalues",            file: T1 + "08-eigenvectors.html", status: "done" },
        { id: "t1-l09", title: "Probability foundations & Bayes' rule", file: T1 + "09-probability-bayes.html", status: "done" },
        { id: "t1-l10", title: "Random variables & distributions",      file: T1 + "10-random-variables.html", status: "done" },
        { id: "t1-l11", title: "Expectation, variance, covariance",     file: T1 + "11-expectation-variance.html", status: "done" },
        { id: "t1-l12", title: "Gaussian, MLE & the bridge to ML",      file: T1 + "12-gaussian-mle.html", status: "done" },
      ],
    },
    {
      n: 2, id: "t2", title: "Optimization",
      desc: "How models learn: gradients, convexity, momentum, Adam, and second-order ideas.",
      lessons: [
        { id: "t2-l00", title: "Derivatives & gradients, geometrically", file: "lessons/track-2-optimization/00-derivatives-gradients.html", status: "done" },
        { id: "t2-l01", title: "Gradient descent from scratch",          file: "lessons/track-2-optimization/01-gradient-descent.html", status: "done" },
        { id: "t2-l02", title: "Convexity & why it matters",             file: "lessons/track-2-optimization/02-convexity.html", status: "done" },
        { id: "t2-l03", title: "Momentum, RMSProp, Adam",                file: "lessons/track-2-optimization/03-momentum-adam.html", status: "done" },
        { id: "t2-l04", title: "Newton's method & second-order ideas",   file: "lessons/track-2-optimization/04-newton-second-order.html", status: "done" },
      ],
    },
    {
      n: 3, id: "t13", title: "Information Theory",
      desc: "The math glue under ML: entropy, cross-entropy, KL divergence, mutual information, and compression.",
      lessons: [
        { id: "t13-l00", title: "Entropy",              file: "lessons/track-13-information-theory/00-entropy.html", status: "done" },
        { id: "t13-l01", title: "Cross-entropy",        file: "lessons/track-13-information-theory/01-cross-entropy.html", status: "done" },
        { id: "t13-l02", title: "KL divergence",        file: "lessons/track-13-information-theory/02-kl-divergence.html", status: "done" },
        { id: "t13-l03", title: "Mutual information",   file: "lessons/track-13-information-theory/03-mutual-information.html", status: "done" },
        { id: "t13-l04", title: "Compression",          file: "lessons/track-13-information-theory/04-compression.html", status: "done" },
      ],
    },
    {
      n: 4, id: "t3", title: "ML Theory",
      desc: "Why learning generalizes: bias–variance, regularization, and PAC intuition.",
      lessons: [
        { id: "t3-l00", title: "The learning problem & generalization", file: "lessons/track-3-ml-theory/00-generalization.html", status: "done" },
        { id: "t3-l01", title: "Bias–variance decomposition",          file: "lessons/track-3-ml-theory/01-bias-variance.html", status: "done" },
        { id: "t3-l02", title: "Regularization (L1/L2) geometrically",  file: "lessons/track-3-ml-theory/02-regularization.html", status: "done" },
        { id: "t3-l03", title: "PAC learning & VC dimension intuition", file: "lessons/track-3-ml-theory/03-pac-vc.html", status: "done" },
      ],
    },
    {
      n: 5, id: "t12", title: "Classical ML",
      desc: "The tabular-data workhorses: k-NN, trees, random forests, gradient boosting, and SVMs.",
      lessons: [
        { id: "t12-l00", title: "Linear & Logistic Regression", file: "lessons/track-12-classical-ml/00-linear-logistic-regression.html", status: "done" },
        { id: "t12-l01", title: "k-Nearest Neighbors",     file: "lessons/track-12-classical-ml/01-knn.html", status: "done" },
        { id: "t12-l02", title: "Decision trees",          file: "lessons/track-12-classical-ml/02-decision-trees.html", status: "done" },
        { id: "t12-l03", title: "Random forests",          file: "lessons/track-12-classical-ml/03-random-forests.html", status: "done" },
        { id: "t12-l04", title: "Gradient boosting",       file: "lessons/track-12-classical-ml/04-gradient-boosting.html", status: "done" },
        { id: "t12-l05", title: "SVMs & the kernel trick", file: "lessons/track-12-classical-ml/05-svm.html", status: "done" },
      ],
    },
    {
      n: 6, id: "t4", title: "Deep Learning from Scratch",
      desc: "Build a working autograd engine, backprop, optimizers, and train MLPs.",
      lessons: [
        { id: "t4-l00", title: "The neuron & the computational graph", file: "lessons/track-4-deep-learning/00-neuron-graph.html", status: "done" },
        { id: "t4-l01", title: "Backpropagation by hand",              file: "lessons/track-4-deep-learning/01-backpropagation.html", status: "done" },
        { id: "t4-l02", title: "Build an autograd engine",             file: "lessons/track-4-deep-learning/02-autograd-engine.html", status: "done" },
        { id: "t4-l03", title: "MLPs, activations & initialization",   file: "lessons/track-4-deep-learning/03-mlp-activations.html", status: "done" },
        { id: "t4-l04", title: "Optimizers & training loops",          file: "lessons/track-4-deep-learning/04-training-loop.html", status: "done" },
      ],
    },
    {
      n: 7, id: "t15", title: "Convolutional Networks & Vision",
      desc: "How machines see: convolution, pooling, a CNN built and trained from scratch, augmentation, and transfer learning.",
      lessons: [
        { id: "t15-l00", title: "Convolution from scratch",     file: "lessons/track-15-cnns-vision/00-convolution.html", status: "done" },
        { id: "t15-l01", title: "Pooling & stride",             file: "lessons/track-15-cnns-vision/01-pooling-stride.html", status: "done" },
        { id: "t15-l02", title: "A ConvNet from scratch",       file: "lessons/track-15-cnns-vision/02-convnet.html", status: "done" },
        { id: "t15-l03", title: "Data augmentation",            file: "lessons/track-15-cnns-vision/03-augmentation.html", status: "done" },
        { id: "t15-l04", title: "Transfer learning",           file: "lessons/track-15-cnns-vision/04-transfer-learning.html", status: "done" },
      ],
    },
    {
      n: 8, id: "t16", title: "Sequence Models (RNN/LSTM/GRU)",
      desc: "Memory over time: recurrent nets, the vanishing-gradient problem, LSTMs and GRUs, and encoder–decoder seq2seq.",
      lessons: [
        { id: "t16-l00", title: "Recurrent neural networks",    file: "lessons/track-16-sequence-models/00-rnn.html", status: "done" },
        { id: "t16-l01", title: "Vanishing & exploding gradients", file: "lessons/track-16-sequence-models/01-vanishing-gradients.html", status: "done" },
        { id: "t16-l02", title: "LSTM: gated memory",           file: "lessons/track-16-sequence-models/02-lstm.html", status: "done" },
        { id: "t16-l03", title: "GRU: streamlined gating",      file: "lessons/track-16-sequence-models/03-gru.html", status: "done" },
        { id: "t16-l04", title: "Sequence-to-sequence",         file: "lessons/track-16-sequence-models/04-seq2seq.html", status: "done" },
      ],
    },
    {
      n: 9, id: "t5", title: "Transformers from Scratch",
      desc: "Attention, positional encodings, and a small GPT you implement and train.",
      lessons: [
        { id: "t5-l00", title: "From sequences to attention",   file: "lessons/track-5-transformers/00-sequences-attention.html", status: "done" },
        { id: "t5-l01", title: "Self-attention, step by step",  file: "lessons/track-5-transformers/01-self-attention.html", status: "done" },
        { id: "t5-l02", title: "Multi-head attention & MLP block", file: "lessons/track-5-transformers/02-multihead-block.html", status: "done" },
        { id: "t5-l03", title: "Positional encodings",          file: "lessons/track-5-transformers/03-positional-encoding.html", status: "done" },
        { id: "t5-l04", title: "Build & train a small GPT",     file: "lessons/track-5-transformers/04-build-gpt.html", status: "done" },
      ],
    },
    {
      n: 10, id: "t11", title: "Large Language Models",
      desc: "From toy GPT to modern LLMs: BPE, scaling laws, LoRA, efficient inference, and RAG.",
      lessons: [
        { id: "t11-l00", title: "Tokenization at scale: BPE",        file: "lessons/track-11-llms/00-tokenization-bpe.html", status: "done" },
        { id: "t11-l01", title: "Scaling laws & the training recipe", file: "lessons/track-11-llms/01-scaling-laws.html", status: "done" },
        { id: "t11-l02", title: "Fine-tuning & LoRA",                 file: "lessons/track-11-llms/02-lora-finetuning.html", status: "done" },
        { id: "t11-l03", title: "Efficient inference: KV-cache & quantization", file: "lessons/track-11-llms/03-efficient-inference.html", status: "done" },
        { id: "t11-l04", title: "Retrieval-augmented generation",     file: "lessons/track-11-llms/04-rag.html", status: "done" },
      ],
    },
    {
      n: 11, id: "t6", title: "Generative Models",
      desc: "VAEs, GANs, diffusion, and autoregressive models — theory and code.",
      lessons: [
        { id: "t6-l00", title: "What is a generative model?",  file: "lessons/track-6-generative/00-what-is-generative.html", status: "done" },
        { id: "t6-l01", title: "Autoregressive models",        file: "lessons/track-6-generative/01-autoregressive.html", status: "done" },
        { id: "t6-l02", title: "Variational autoencoders",     file: "lessons/track-6-generative/02-vae.html", status: "done" },
        { id: "t6-l03", title: "GANs",                         file: "lessons/track-6-generative/03-gan.html", status: "done" },
        { id: "t6-l04", title: "Diffusion models",             file: "lessons/track-6-generative/04-diffusion.html", status: "done" },
      ],
    },
    {
      n: 12, id: "t7", title: "Reinforcement Learning",
      desc: "MDPs, value & policy methods, policy gradients, and RLHF.",
      lessons: [
        { id: "t7-l00", title: "MDPs & the RL problem",        file: "lessons/track-7-rl/00-mdps.html", status: "done" },
        { id: "t7-l01", title: "Value iteration & Q-learning", file: "lessons/track-7-rl/01-value-q-learning.html", status: "done" },
        { id: "t7-l02", title: "Policy gradients (REINFORCE)", file: "lessons/track-7-rl/02-policy-gradients.html", status: "done" },
        { id: "t7-l03", title: "Actor–critic & PPO",           file: "lessons/track-7-rl/03-actor-critic-ppo.html", status: "done" },
        { id: "t7-l04", title: "RLHF: aligning language models", file: "lessons/track-7-rl/04-rlhf.html", status: "done" },
      ],
    },
    {
      n: 13, id: "t17", title: "Learning Paradigms",
      desc: "Beyond supervised labels: unsupervised clustering, self-supervised & contrastive learning, semi-supervised, and active learning.",
      lessons: [
        { id: "t17-l00", title: "Learning without labels: k-means", file: "lessons/track-17-learning-paradigms/00-learning-paradigms.html", status: "done" },
        { id: "t17-l01", title: "Self-supervised learning", file: "lessons/track-17-learning-paradigms/01-self-supervised.html", status: "done" },
        { id: "t17-l02", title: "Contrastive learning (SimCLR)", file: "lessons/track-17-learning-paradigms/02-contrastive.html", status: "done" },
        { id: "t17-l03", title: "Semi-supervised learning", file: "lessons/track-17-learning-paradigms/03-semi-supervised.html", status: "done" },
        { id: "t17-l04", title: "Active learning", file: "lessons/track-17-learning-paradigms/04-active-learning.html", status: "done" },
      ],
    },
    {
      n: 14, id: "t20", title: "Graph Neural Networks",
      desc: "Learning on graphs: message passing, GCNs and GATs from scratch, and node/graph tasks.",
      lessons: [
        { id: "t20-l00", title: "Graphs as data", file: "lessons/track-20-graph-neural-networks/00-graphs-as-data.html", status: "done" },
        { id: "t20-l01", title: "Message passing", file: "lessons/track-20-graph-neural-networks/01-message-passing.html", status: "done" },
        { id: "t20-l02", title: "Graph convolutions (GCN)", file: "lessons/track-20-graph-neural-networks/02-gcn.html", status: "done" },
        { id: "t20-l03", title: "Attention on graphs (GAT)", file: "lessons/track-20-graph-neural-networks/03-gat.html", status: "done" },
        { id: "t20-l04", title: "Graph classification", file: "lessons/track-20-graph-neural-networks/04-graph-classification.html", status: "done" },
      ],
    },
    {
      n: 15, id: "t21", title: "Probabilistic & Bayesian ML",
      desc: "Reasoning with uncertainty: Bayesian inference, MCMC, variational inference, Gaussian processes, and calibration.",
      lessons: [
        { id: "t21-l00", title: "Bayesian inference", file: "lessons/track-21-bayesian-ml/00-bayesian-inference.html", status: "done" },
        { id: "t21-l01", title: "MCMC sampling", file: "lessons/track-21-bayesian-ml/01-mcmc.html", status: "done" },
        { id: "t21-l02", title: "Variational inference", file: "lessons/track-21-bayesian-ml/02-variational-inference.html", status: "done" },
        { id: "t21-l03", title: "Gaussian processes", file: "lessons/track-21-bayesian-ml/03-gaussian-processes.html", status: "done" },
        { id: "t21-l04", title: "Uncertainty & calibration", file: "lessons/track-21-bayesian-ml/04-calibration.html", status: "done" },
      ],
    },
    {
      n: 16, id: "t19", title: "Meta, Few-Shot & Continual Learning",
      desc: "Learning to learn and learning without forgetting: MAML, prototypical nets, and continual learning.",
      lessons: [
        { id: "t19-l00", title: "Few-shot learning", file: "lessons/track-19-meta-learning/00-few-shot-learning.html", status: "done" },
        { id: "t19-l01", title: "MAML (learning to learn)", file: "lessons/track-19-meta-learning/01-maml.html", status: "done" },
        { id: "t19-l02", title: "Prototypical networks", file: "lessons/track-19-meta-learning/02-prototypical-networks.html", status: "done" },
        { id: "t19-l03", title: "Catastrophic forgetting", file: "lessons/track-19-meta-learning/03-catastrophic-forgetting.html", status: "done" },
        { id: "t19-l04", title: "Continual learning (EWC/replay)", file: "lessons/track-19-meta-learning/04-continual-learning.html", status: "done" },
      ],
    },
    {
      n: 17, id: "t18", title: "Federated & Privacy-Preserving ML",
      desc: "Train without pooling data: federated averaging, non-IID data, differential privacy, and DP-SGD.",
      lessons: [
        { id: "t18-l00", title: "Federated averaging (FedAvg)", file: "lessons/track-18-federated-privacy/00-federated-averaging.html", status: "done" },
        { id: "t18-l01", title: "The non-IID challenge", file: "lessons/track-18-federated-privacy/01-non-iid.html", status: "done" },
        { id: "t18-l02", title: "Differential privacy", file: "lessons/track-18-federated-privacy/02-differential-privacy.html", status: "done" },
        { id: "t18-l03", title: "DP-SGD", file: "lessons/track-18-federated-privacy/03-dp-sgd.html", status: "done" },
        { id: "t18-l04", title: "Secure aggregation", file: "lessons/track-18-federated-privacy/04-secure-aggregation.html", status: "done" },
      ],
    },
    {
      n: 18, id: "t22", title: "Causal Inference",
      desc: "Beyond correlation: confounding, the do-operator, backdoor adjustment, and treatment effects.",
      lessons: [
        { id: "t22-l00", title: "Correlation vs causation", file: "lessons/track-22-causal-inference/00-correlation-vs-causation.html", status: "done" },
        { id: "t22-l01", title: "Confounding", file: "lessons/track-22-causal-inference/01-confounding.html", status: "done" },
        { id: "t22-l02", title: "The do-operator", file: "lessons/track-22-causal-inference/02-the-do-operator.html", status: "done" },
        { id: "t22-l03", title: "Backdoor adjustment", file: "lessons/track-22-causal-inference/03-backdoor-adjustment.html", status: "done" },
        { id: "t22-l04", title: "Estimating treatment effects", file: "lessons/track-22-causal-inference/04-estimating-treatment-effects.html", status: "done" },
      ],
    },
    {
      n: 19, id: "t23", title: "Efficient & Scalable ML",
      desc: "Making models small and training fast: distillation, pruning, quantization, and parallelism.",
      lessons: [
        { id: "t23-l00", title: "Knowledge distillation", file: "lessons/track-23-efficient-ml/00-knowledge-distillation.html", status: "done" },
        { id: "t23-l01", title: "Pruning", file: "lessons/track-23-efficient-ml/01-pruning.html", status: "done" },
        { id: "t23-l02", title: "Quantization", file: "lessons/track-23-efficient-ml/02-quantization.html", status: "done" },
        { id: "t23-l03", title: "Data & model parallelism", file: "lessons/track-23-efficient-ml/03-parallelism.html", status: "done" },
        { id: "t23-l04", title: "Mixed precision", file: "lessons/track-23-efficient-ml/04-mixed-precision.html", status: "done" },
      ],
    },
    {
      n: 20, id: "t24", title: "Frontier Architectures & Agents",
      desc: "The cutting edge: mixture-of-experts, state-space models, multimodal models, and LLM agents.",
      lessons: [
        { id: "t24-l00", title: "Mixture of Experts", file: "lessons/track-24-frontier-agents/00-mixture-of-experts.html", status: "done" },
        { id: "t24-l01", title: "State-space models (Mamba)", file: "lessons/track-24-frontier-agents/01-state-space-models.html", status: "done" },
        { id: "t24-l02", title: "Multimodal models (CLIP)", file: "lessons/track-24-frontier-agents/02-multimodal-models.html", status: "done" },
        { id: "t24-l03", title: "LLM agents & tool use", file: "lessons/track-24-frontier-agents/03-llm-agents.html", status: "done" },
        { id: "t24-l04", title: "Planning & memory", file: "lessons/track-24-frontier-agents/04-planning-and-memory.html", status: "done" },
      ],
    },
    {
      n: 21, id: "t8", title: "Reading & Reproducing Papers",
      desc: "How to dissect, reproduce, and critique research papers.",
      lessons: [
        { id: "t8-l00", title: "How to read a paper (3-pass method)", file: "lessons/track-8-papers/00-how-to-read.html", status: "done" },
        { id: "t8-l01", title: "Reproducing results faithfully",      file: "lessons/track-8-papers/01-reproducing.html", status: "done" },
        { id: "t8-l02", title: "Critiquing claims & evidence",        file: "lessons/track-8-papers/02-critiquing.html", status: "done" },
      ],
    },
    {
      n: 22, id: "t9", title: "Research Methods",
      desc: "Experiment design, ablations, honest evaluation, and writing up results.",
      lessons: [
        { id: "t9-l00", title: "Designing experiments",     file: "lessons/track-9-methods/00-designing-experiments.html", status: "done" },
        { id: "t9-l01", title: "Ablations & controls",      file: "lessons/track-9-methods/01-ablations.html", status: "done" },
        { id: "t9-l02", title: "Honest evaluation & pitfalls", file: "lessons/track-9-methods/02-honest-evaluation.html", status: "done" },
        { id: "t9-l03", title: "Writing the paper",         file: "lessons/track-9-methods/03-writing.html", status: "done" },
      ],
    },
    {
      n: 23, id: "t14", title: "AI Safety & Interpretability",
      desc: "Alignment, reward hacking, mechanistic interpretability, evaluation & red-teaming, adversarial robustness.",
      lessons: [
        { id: "t14-l00", title: "The alignment problem",       file: "lessons/track-14-safety/00-alignment-problem.html", status: "done" },
        { id: "t14-l01", title: "Goodhart & reward hacking",   file: "lessons/track-14-safety/01-goodhart-reward-hacking.html", status: "done" },
        { id: "t14-l02", title: "Mechanistic interpretability", file: "lessons/track-14-safety/02-interpretability.html", status: "done" },
        { id: "t14-l03", title: "Evaluations & red-teaming",   file: "lessons/track-14-safety/03-evaluations.html", status: "done" },
        { id: "t14-l04", title: "Adversarial robustness",      file: "lessons/track-14-safety/04-adversarial-robustness.html", status: "done" },
      ],
    },
    {
      n: 24, id: "t10", title: "Research Capstones",
      desc: "Projects that actually run: reproduce a classic, match a result, run an honest novel experiment.",
      lessons: [
        { id: "t10-l00", title: "Capstone A: ship a from-scratch GPT", file: "lessons/track-10-capstones/00-capstone-namegen.html", status: "done" },
        { id: "t10-l01", title: "Capstone B: reproduce a classic, live", file: "lessons/track-10-capstones/01-capstone-ddlab.html", status: "done" },
        { id: "t10-l02", title: "Capstone C: a live RL agent", file: "lessons/track-10-capstones/02-capstone-gridworld.html", status: "done" },
      ],
    },
  
  ];

  /* expose for any page that wants it */
  window.CURRICULUM = TRACKS;

  /* --------------------------------------------------------------------------
     helpers
     -------------------------------------------------------------------------- */
  const ROOT = document.documentElement.dataset.root || "";
  const CURRENT = document.documentElement.dataset.lesson || null;
  const PAGE = document.body.dataset.page || "lesson";
  try { window.GA_TRACKS = TRACKS; } catch (e) {}
  const $ = (s, el) => (el || document).querySelector(s);
  const $$ = (s, el) => Array.prototype.slice.call((el || document).querySelectorAll(s));
  const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };

  const store = {
    get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : v; } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (e) {} },
  };
  const isDone = (id) => store.get("done:" + id) === "1";

  // flat ordered list of *built* lessons -> drives prev/next
  const FLAT_DONE = [];
  TRACKS.forEach((t) => t.lessons.forEach((l) => { if (l.status === "done") FLAT_DONE.push(Object.assign({ track: t }, l)); }));

  /* --------------------------------------------------------------------------
     sidebar
     -------------------------------------------------------------------------- */
  function buildSidebar() {
    const sb = $("#sidebar");
    if (!sb) return;
    const theme = document.documentElement.getAttribute("data-theme") || "light";

    const head = el("div", "sb-head");
    head.innerHTML =
      '<a class="sb-brand" href="' + ROOT + 'researcher-path.html" style="text-decoration:none;color:inherit">' +
        '<span class="logo"><svg viewBox="0 0 40 40" fill="none" aria-hidden="true"><path d="M12 26h16L20 13z" stroke="#fff" stroke-width="3.6" stroke-linejoin="round"/><circle cx="20" cy="8.6" r="2.1" fill="#fff"/></svg></span>' +
        '<span><span class="t">Gradient Ascent</span><br><span class="s">first principles → frontier</span></span>' +
      '</a>' +
      '<div class="sb-tools">' +
        '<a class="sb-home" href="' + ROOT + 'researcher-path.html">⌂ Home</a>' +
        '<button id="themeBtn" title="Toggle light/dark">' + (theme === "dark" ? "☀︎ Light" : "☾ Dark") + '</button>' +
      '</div>';
    sb.appendChild(head);

    const onramp = el("div");
    onramp.style.cssText = "display:flex;flex-wrap:wrap;gap:.3rem;padding:.5rem .9rem .2rem;";
    function _oc(href, label, primary) {
      return '<a href="' + ROOT + href + '" style="flex:1 1 auto;text-align:center;white-space:nowrap;font-size:.75rem;font-weight:600;text-decoration:none;border-radius:8px;padding:.4rem .5rem;' +
        (primary ? 'color:var(--primary-ink);background:var(--primary-weak)' : 'color:var(--text-muted);background:var(--bg);border:1px solid var(--border)') + '">' + label + '</a>';
    }
    onramp.innerHTML = _oc("start-here.html", "\u{1F680} Start", true) + _oc("math-primer.html", "\u2211 Math") +
      _oc("progress.html", "\u{1F4CA} Progress") + _oc("glossary.html", "\u{1F4D6} Glossary") + _oc("flashcards.html", "\u{1F0CF} Cards");
    sb.appendChild(onramp);

    const nav = el("nav", "sb-nav");
    TRACKS.forEach((t) => {
      const det = el("details", "sb-track");
      const hasCurrent = t.lessons.some((l) => l.id === CURRENT);
      if (hasCurrent || (PAGE === "home" && t.n === 1)) det.open = true;
      const sum = el("summary");
      sum.innerHTML = '<span class="tnum">' + t.n + '</span><span>' + t.title + '</span><span class="tchev">▸</span>';
      det.appendChild(sum);

      const ul = el("ul", "sb-lessons");
      t.lessons.forEach((l) => {
        const li = el("li");
        if (l.status === "done") {
          const a = el("a");
          a.href = ROOT + l.file;
          if (l.id === CURRENT) a.classList.add("active");
          const check = isDone(l.id) ? '<span class="ldone" data-check="' + l.id + '">✓</span>' : '<span class="ldone" data-check="' + l.id + '" style="visibility:hidden">✓</span>';
          a.innerHTML = check + "<span>" + l.title + "</span>";
          li.appendChild(a);
        } else {
          const s = el("span", "locked");
          s.innerHTML = "<span>" + l.title + "</span><span class='badge-soon'>soon</span>";
          li.appendChild(s);
        }
        ul.appendChild(li);
      });
      det.appendChild(ul);
      nav.appendChild(det);
    });
    sb.appendChild(nav);

    const tb = $("#themeBtn");
    if (tb) tb.addEventListener("click", toggleTheme);
  }

  /* --------------------------------------------------------------------------
     breadcrumb + prev/next  (full page nav => browser lands at top)
     -------------------------------------------------------------------------- */
  function pagerLink(l, dir) {
    if (!l) {
      const label = dir === "next" ? "More lessons coming soon" : "You're at the start";
      const a = el("a", "disabled " + dir);
      a.innerHTML = '<span class="dir">' + (dir === "next" ? "Next →" : "← Previous") + '</span><span class="ttl">' + label + "</span>";
      return a;
    }
    const a = el("a", dir);
    a.href = ROOT + l.file;
    a.innerHTML = '<span class="dir">' + (dir === "next" ? "Next →" : "← Previous") + '</span><span class="ttl">' + l.title + "</span>";
    return a;
  }

  function buildLessonChrome() {
    const art = $("article.lesson");
    if (!art || !CURRENT) return;
    const idx = FLAT_DONE.findIndex((l) => l.id === CURRENT);
    const prev = idx > 0 ? FLAT_DONE[idx - 1] : null;
    const next = idx >= 0 && idx < FLAT_DONE.length - 1 ? FLAT_DONE[idx + 1] : null;
    const me = FLAT_DONE[idx];

    // breadcrumb
    const crumb = el("div", "crumb");
    crumb.innerHTML =
      '<a href="' + ROOT + 'researcher-path.html">Home</a> &nbsp;›&nbsp; ' +
      '<a href="' + ROOT + 'researcher-path.html">Track ' + (me ? me.track.n : 1) + " · " + (me ? me.track.title : "") + "</a>";
    art.insertBefore(crumb, art.firstChild);

    // top pager (compact)
    const top = el("div", "pager top");
    top.appendChild(pagerLink(prev, "prev"));
    top.appendChild(pagerLink(next, "next"));
    // place top pager after the meta-row if present, else after subtitle, else after h1
    const anchor = $(".meta-row", art) || $(".subtitle", art) || $("h1", art);
    if (anchor && anchor.nextSibling) anchor.parentNode.insertBefore(top, anchor.nextSibling);
    else art.appendChild(top);

    // bottom: complete button + pager
    const foot = el("div");
    const crow = el("div", "complete-row");
    const btn = el("button", "btn-complete" + (isDone(CURRENT) ? " done" : ""));
    btn.textContent = isDone(CURRENT) ? "✓ Completed — nice work" : "Mark this lesson complete";
    btn.addEventListener("click", () => {
      const now = !isDone(CURRENT);
      store.set("done:" + CURRENT, now ? "1" : "0");
      if (now) { try { var _d = new Date().toISOString().slice(0, 10); var _days = JSON.parse(store.get("studydays", "[]")); if (_days.indexOf(_d) < 0) { _days.push(_d); store.set("studydays", JSON.stringify(_days)); } } catch (e) {} }
      btn.classList.toggle("done", now);
      btn.textContent = now ? "✓ Completed — nice work" : "Mark this lesson complete";
      const chk = $('[data-check="' + CURRENT + '"]');
      if (chk) chk.style.visibility = now ? "visible" : "hidden";
    });
    crow.appendChild(btn);
    foot.appendChild(crow);

    const bottom = el("div", "pager");
    bottom.appendChild(pagerLink(prev, "prev"));
    bottom.appendChild(pagerLink(next, "next"));
    foot.appendChild(bottom);

    const ft = el("div", "foot");
    ft.innerHTML = "Gradient Ascent · a self-paced path from first principles to the research frontier. " +
      "The test for this lesson: <em>could you now derive it, implement it, and explain it to a peer?</em>";
    foot.appendChild(ft);

    art.appendChild(foot);
  }

  /* --------------------------------------------------------------------------
     quiz engine
     -------------------------------------------------------------------------- */
  function initQuizzes() {
    $$(".quiz-q").forEach((q) => {
      const correct = parseInt(q.getAttribute("data-correct"), 10);
      const ul = $(".quiz-options", q);
      if (!ul) return;
      const opts = $$("li", ul);
      opts.forEach((li, i) => {
        const key = String.fromCharCode(65 + i);
        const k = el("span", "opt-key", key + ".");
        li.insertBefore(k, li.firstChild);
        li.addEventListener("click", () => {
          if (ul.classList.contains("answered")) return;
          ul.classList.add("answered");
          opts.forEach((o, j) => {
            o.classList.add("reveal");
            if (j === correct) o.classList.add("correct");
          });
          if (i !== correct) li.classList.add("wrong");
          // result tag, placed just above the options list
          const tag = el("div", "quiz-tag " + (i === correct ? "ok" : "no"), i === correct ? "✓ Correct" : "✗ Not quite — see why below");
          q.insertBefore(tag, ul);
        });
      });
    });
  }

  /* --------------------------------------------------------------------------
     tiny, safe Python syntax highlighter
     -------------------------------------------------------------------------- */
  function highlight() {
    const KW = "def|class|return|import|from|as|if|elif|else|for|while|in|not|and|or|None|True|False|with|lambda|yield|pass|break|continue|global|nonlocal|try|except|finally|raise|assert|is|del|print|self|async|await";
    const re = new RegExp(
      "(#[^\\n]*)" +                                   // 1 comment
      "|('(?:[^'\\\\]|\\\\.)*'|\"(?:[^\"\\\\]|\\\\.)*\")" + // 2 string
      "|\\b(\\d+\\.?\\d*)\\b" +                        // 3 number
      "|\\b(" + KW + ")\\b",                           // 4 keyword
      "g"
    );
    $$("pre code").forEach((code) => {
      if (code.dataset.nohl != null) return;
      try {
        let txt = code.textContent;
        txt = txt.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
        txt = txt.replace(re, (m, c, s, n, k) => {
          if (c) return '<span class="tok-com">' + c + "</span>";
          if (s) return '<span class="tok-str">' + s + "</span>";
          if (n) return '<span class="tok-num">' + n + "</span>";
          if (k) return '<span class="tok-kw">' + k + "</span>";
          return m;
        });
        code.innerHTML = txt;
      } catch (e) { /* leave plain on any error */ }
    });
  }

  /* --------------------------------------------------------------------------
     KaTeX
     -------------------------------------------------------------------------- */
  function renderMath() {
    if (!window.renderMathInElement) return;
    try {
      window.renderMathInElement(document.body, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "\\[", right: "\\]", display: true },
          { left: "$", right: "$", display: false },
          { left: "\\(", right: "\\)", display: false },
        ],
        throwOnError: false,
        macros: { "\\R": "\\mathbb{R}", "\\E": "\\mathbb{E}", "\\norm": "\\left\\lVert#1\\right\\rVert", "\\vv": "\\mathbf{#1}" },
      });
    } catch (e) {}
  }

  /* --------------------------------------------------------------------------
     theme, progress bar, mobile menu, home grid
     -------------------------------------------------------------------------- */
  function applyTheme(t) {
    document.documentElement.setAttribute("data-theme", t);
    store.set("theme", t);
    const b = $("#themeBtn");
    if (b) b.innerHTML = t === "dark" ? "☀︎ Light" : "☾ Dark";
  }
  function toggleTheme() {
    const cur = document.documentElement.getAttribute("data-theme") || "light";
    applyTheme(cur === "dark" ? "light" : "dark");
  }

  function progressBar() {
    let bar = $("#progress-bar");
    if (!bar) { bar = el("div"); bar.id = "progress-bar"; document.body.appendChild(bar); }
    const upd = () => {
      const h = document.documentElement.scrollHeight - window.innerHeight;
      const p = h > 0 ? (window.scrollY / h) * 100 : 0;
      bar.style.width = p + "%";
    };
    window.addEventListener("scroll", upd, { passive: true });
    upd();
  }

  function mobileMenu() {
    let btn = $(".nav-toggle");
    if (!btn) { btn = el("button", "nav-toggle", "☰"); btn.setAttribute("aria-label", "Menu"); document.body.appendChild(btn); }
    let ov = $("#overlay");
    if (!ov) { ov = el("div"); ov.id = "overlay"; document.body.appendChild(ov); }
    const sb = $("#sidebar");
    const close = () => { sb && sb.classList.remove("open"); ov.classList.remove("show"); };
    btn.addEventListener("click", () => { sb && sb.classList.toggle("open"); ov.classList.toggle("show"); });
    ov.addEventListener("click", close);
  }

  function buildHome() {
    const grid = $("#track-grid");
    if (!grid) return;
    TRACKS.forEach((t) => {
      const done = t.lessons.filter((l) => l.status === "done").length;
      const total = t.lessons.length;
      const ready = done > 0;
      const firstDone = t.lessons.find((l) => l.status === "done");
      const href = firstDone ? ROOT + firstDone.file : "#";
      const card = el(ready ? "a" : "div", "tcard");
      if (ready) card.href = href;
      const pct = Math.round((done / total) * 100);
      card.innerHTML =
        '<div class="tc-top"><span class="tc-num">' + t.n + '</span><h3>' + t.title + "</h3></div>" +
        '<div class="tc-desc">' + t.desc + "</div>" +
        '<div class="tprog"><span style="width:' + pct + '%"></span></div>' +
        '<div class="tc-foot"><span class="tc-stat">' + done + " / " + total + " lessons" + (ready ? "" : " planned") + "</span>" +
        '<span class="pill ' + (ready ? "ready" : "soon") + '">' + (ready ? "Start \u25B8" : "Coming soon") + "</span></div>";
      grid.appendChild(card);
    });
  }

  /* --------------------------------------------------------------------------
     boot
     -------------------------------------------------------------------------- */
  /* --------------------------------------------------------------------------
     in-browser Python playground: ▶ Run on every code block (lazy Pyodide)
     -------------------------------------------------------------------------- */
  var __pyReady = null;
  function __loadPy() {
    if (__pyReady) return __pyReady;
    __pyReady = new Promise(function (res, rej) {
      var sc = document.createElement("script");
      sc.src = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js";
      sc.onload = function () {
        try {
          window.loadPyodide().then(function (py) {
            return py.loadPackage("numpy").then(function () { res(py); });
          }).catch(rej);
        } catch (e) { rej(e); }
      };
      sc.onerror = function () { rej(new Error("Could not load Python (Pyodide). Check your connection and try again.")); };
      document.head.appendChild(sc);
    });
    return __pyReady;
  }
  function enhanceCode() {
    var main = document.querySelector("main"); if (!main) return;
    var blocks = main.querySelectorAll("pre > code"); if (!blocks.length) return;
    if (!document.getElementById("code-play-css")) {
      var st = document.createElement("style"); st.id = "code-play-css";
      st.textContent =
        ".code-h .code-actions{margin-left:auto;display:flex;gap:.4rem;align-items:center}" +
        ".code-run,.code-edit{font:600 .72rem var(--sans);text-transform:none;letter-spacing:0;cursor:pointer;border:1px solid var(--border-strong);background:var(--surface);color:var(--text);border-radius:6px;padding:.22rem .62rem;line-height:1.15}" +
        ".code-run{background:var(--primary);color:#fff;border-color:transparent}" +
        ".code-run:hover{filter:brightness(1.06)}.code-run:disabled{opacity:.55;cursor:default}" +
        ".code-edit:hover{border-color:var(--primary);color:var(--primary-ink)}" +
        ".code-edit.on{background:var(--primary-weak);color:var(--primary-ink);border-color:transparent}" +
        ".code-outwrap{display:none;margin:.55rem 0 1.4rem}" +
        ".code-outh{display:flex;align-items:center;gap:.5rem;font:600 .72rem var(--sans);text-transform:uppercase;letter-spacing:.4px;color:var(--text-faint);margin:0 0 .35rem;padding-left:.2rem}" +
        ".code-outh .odot{width:9px;height:9px;border-radius:50%;background:var(--green)}" +
        ".code-outh.busy .odot{background:var(--amber)}.code-outh.err .odot{background:var(--red)}" +
        ".code-out{margin:0;background:var(--code-bg);color:var(--code-text);border-radius:var(--radius-sm);padding:.8rem 1rem;white-space:pre-wrap;font:.82rem/1.55 var(--mono);max-height:340px;overflow:auto;border:1px solid #1e293b}" +
        ".code-out.err{color:#fca5a5}" +
        "code.codeedit{outline:2px solid var(--primary);outline-offset:4px;border-radius:5px}";
      document.head.appendChild(st);
    }
    var reIsPy = /(^|\n)\s*(import |from |def |class |print\(|np\.|for |while |if |return |#|[A-Za-z_]\w*\s*=)/;
    blocks.forEach(function (code) {
      var pre = code.parentElement;
      if (pre.getAttribute("data-play") === "1") return;
      if (pre.getAttribute("data-run") !== "1") return;
      var prev = pre.previousElementSibling;
      if (prev && prev.classList && prev.classList.contains("code-h") && prev.classList.contains("out")) return;
      var txt = code.innerText || "";
      if (txt.trim().length < 3 || !reIsPy.test(txt)) return;
      pre.setAttribute("data-play", "1");
      var header;
      if (prev && prev.classList && prev.classList.contains("code-h")) {
        header = prev;
      } else {
        header = document.createElement("div"); header.className = "code-h";
        header.innerHTML = '<span class="dot"></span> python';
        pre.parentNode.insertBefore(header, pre);
      }
      var actions = document.createElement("span"); actions.className = "code-actions";
      var runBtn = document.createElement("button"); runBtn.className = "code-run"; runBtn.type = "button"; runBtn.textContent = "▶ Run";
      runBtn.title = "Run this Python in your browser — numpy included";
      var editBtn = document.createElement("button"); editBtn.className = "code-edit"; editBtn.type = "button"; editBtn.textContent = "✎ Edit";
      editBtn.title = "Edit the code, then Run";
      actions.appendChild(runBtn); actions.appendChild(editBtn); header.appendChild(actions);
      var outwrap = document.createElement("div"); outwrap.className = "code-outwrap";
      outwrap.innerHTML = '<div class="code-outh"><span class="odot"></span> output</div>';
      var out = document.createElement("pre"); out.className = "code-out"; outwrap.appendChild(out);
      pre.parentNode.insertBefore(outwrap, pre.nextSibling);
      editBtn.addEventListener("click", function () {
        var on = code.contentEditable !== "true";
        code.contentEditable = on ? "true" : "false";
        code.classList.toggle("codeedit", on);
        editBtn.classList.toggle("on", on);
        editBtn.textContent = on ? "✎ Editing" : "✎ Edit";
        if (on) code.focus();
      });
      runBtn.addEventListener("click", function () { runCode(code.innerText, outwrap, out, runBtn); });
    });
    function runCode(src, outwrap, out, btn) {
      if (/\bnp\b/.test(src) && !/\bimport\s+numpy\b/.test(src) && !/\bas\s+np\b/.test(src)) src = "import numpy as np\n" + src;
      outwrap.style.display = "block";
      var oh = outwrap.querySelector(".code-outh");
      oh.className = "code-outh busy"; out.className = "code-out";
      out.textContent = "Loading Python… the first run downloads it once (~10 MB), then it is instant.";
      btn.disabled = true; var label = btn.textContent; btn.textContent = "… running";
      __loadPy().then(function (py) {
        var buf = "";
        py.setStdout({ batched: function (s) { buf += s + "\n"; } });
        py.setStderr({ batched: function (s) { buf += s + "\n"; } });
        return py.runPythonAsync(src).then(function () {
          oh.className = "code-outh"; out.className = "code-out";
          out.textContent = buf.trim() === "" ? "✓ ran — no printed output. Add print(...) to see values." : buf;
        }).catch(function (e) {
          oh.className = "code-outh err"; out.className = "code-out err";
          out.textContent = (buf ? buf + "\n" : "") + String((e && e.message) || e);
        });
      }).catch(function (e) {
        oh.className = "code-outh err"; out.className = "code-out err";
        out.textContent = "⚠ " + String((e && e.message) || e);
      }).then(function () { btn.disabled = false; btn.textContent = label; });
    }
  }

  /* --------------------------------------------------------------------------
     read-aloud walkthrough: browser TTS narrates the lesson, scrolling +
     highlighting each part as it goes (lessons only; no server, no API key)
     -------------------------------------------------------------------------- */
  function readAloud() {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
    if (!document.documentElement.dataset.lesson) return;
    var main = document.querySelector("main"); if (!main) return;
    var synth = window.speechSynthesis;
    var st = document.createElement("style");
    st.textContent =
      ".rd-bar{position:fixed;right:16px;bottom:16px;z-index:60;display:flex;gap:.35rem;align-items:center;background:var(--surface);border:1px solid var(--border);border-radius:999px;padding:.32rem .4rem;box-shadow:var(--shadow-lg)}" +
      ".rd-bar button{font:600 .82rem var(--sans);cursor:pointer;border:none;background:transparent;color:var(--text);border-radius:999px;padding:.36rem .7rem}" +
      ".rd-bar .rd-toggle{background:var(--primary);color:#fff}" +
      ".rd-bar .rd-stop,.rd-bar .rd-speed{background:var(--bg);border:1px solid var(--border)}" +
      ".rd-bar button:hover{filter:brightness(1.06)}" +
      ".rd-on{background:var(--primary-weak);box-shadow:0 0 0 4px var(--primary-weak);border-radius:6px}" +
      "@media print{.rd-bar{display:none}}";
    document.head.appendChild(st);
    var bar = document.createElement("div"); bar.className = "rd-bar";
    bar.innerHTML = '<button class="rd-toggle" type="button" title="Read this lesson aloud">🔊 Listen</button>' +
      '<button class="rd-stop" type="button" title="Stop" hidden>■</button>' +
      '<button class="rd-speed" type="button" title="Speed" hidden>1×</button>';
    document.body.appendChild(bar);
    var toggle = bar.querySelector(".rd-toggle"), stopB = bar.querySelector(".rd-stop"), speedB = bar.querySelector(".rd-speed");
    var queue = [], idx = 0, curEl = null, playing = false, paused = false;
    var speeds = [1, 1.25, 1.5, 0.85], si = 0, rate = 1, voice = null;
    function pickVoice() { var vs = synth.getVoices(); voice = vs.filter(function (v) { return /en[-_]/i.test(v.lang); })[0] || vs[0] || null; }
    pickVoice(); synth.onvoiceschanged = pickVoice;
    function sentences(t) { return t.match(/[^.!?;:]+[.!?;:]+|[^.!?;:]+$/g) || [t]; }
    function clean(el) {
      var c = el.cloneNode(true);
      c.querySelectorAll(".katex, .katex-display, .katex-mathml").forEach(function (k) { k.replaceWith(document.createTextNode(" ")); });
      c.querySelectorAll("pre, code, script, style").forEach(function (k) { k.remove(); });
      return (c.textContent || "").replace(/\s+/g, " ").trim();
    }
    function build() {
      queue = [];
      var wrap = main.querySelector("article") || main;
      var nodes = [].slice.call(wrap.querySelectorAll("h1, h2, h3, p, li, figcaption, .co-h, .rk"));
      nodes.forEach(function (el) {
        if (el.closest(".widget, .quiz, pre, script, style, .rd-bar, .deepdive")) return;
        var t = clean(el); if (t.length < 2) return;
        sentences(t).forEach(function (se) { se = se.trim(); if (se.length) queue.push({ el: el, text: se }); });
      });
    }
    function hl(el) {
      if (curEl === el) return;
      if (curEl) curEl.classList.remove("rd-on");
      curEl = el;
      if (el) { el.classList.add("rd-on"); try { el.scrollIntoView({ behavior: "smooth", block: "center" }); } catch (e) {} }
    }
    function speakNext() {
      if (!playing) return;
      if (idx >= queue.length) { finish(); return; }
      var item = queue[idx]; hl(item.el);
      var u = new SpeechSynthesisUtterance(item.text); u.rate = rate; u.lang = "en-US";
      if (voice) u.voice = voice;
      u.onend = function () { if (!playing) return; idx++; speakNext(); };
      u.onerror = function () { if (!playing) return; idx++; speakNext(); };
      synth.speak(u);
    }
    function start() { build(); if (!queue.length) return; idx = 0; playing = true; paused = false; synth.cancel(); toggle.textContent = "⏸ Pause"; stopB.hidden = false; speedB.hidden = false; speakNext(); }
    function finish() { playing = false; paused = false; synth.cancel(); hl(null); toggle.textContent = "🔊 Listen"; stopB.hidden = true; speedB.hidden = true; }
    toggle.addEventListener("click", function () {
      if (!playing) { start(); }
      else if (paused) { synth.resume(); paused = false; toggle.textContent = "⏸ Pause"; }
      else { synth.pause(); paused = true; toggle.textContent = "▶ Resume"; }
    });
    stopB.addEventListener("click", finish);
    speedB.addEventListener("click", function () { si = (si + 1) % speeds.length; rate = speeds[si]; speedB.textContent = rate + "×"; if (playing && !paused) { synth.cancel(); speakNext(); } });
    setInterval(function () { if (playing && !paused && synth.speaking) synth.resume(); }, 9000);
    window.addEventListener("beforeunload", function () { synth.cancel(); });
  }

  function boot() {
    // theme: stored or system preference
    const saved = store.get("theme", null);
    const sys = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", saved || sys);

    buildSidebar();
    buildLessonChrome();
    initQuizzes();
    highlight();
    renderMath();
    enhanceCode();
    progressBar();
    mobileMenu();
    buildHome();
    readAloud();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();

## Rethinking MTP Through Real-World Analogies: Better & Faster Large Language Models via Multi-Token Prediction (Gloeckle et al., 2024)

- Paper: https://ar5iv.labs.arxiv.org/html/2404.19737

> What follows is not a summary of the paper. These are my own thoughts, analogies, and brainstorm ideas that came to me while studying it.
> **All insights are exploratory and unverified** — they represent directions I find worth exploring, not claims backed by experiments or evidence.


### Insight 1: Reading Dense Papers vs. Reading Novels — Why Difficulty Is the True Engine of Emergence

Something struck me as strange when I first read this paper: the next-token prediction objective is almost absurdly simple — look at everything before, guess the next word. That one task, and yet models somehow learn grammar, common sense, and even basic reasoning. That's already a miracle. So why do we need MTP at all?

**Analogy:** Imagine two very different reading experiences. When you read a thriller novel, you're passive. The plot carries you forward. Your brain doesn't have to do much. But when you sit down with a research paper in an unfamiliar field, something completely different happens. You hit a concept and stop. You ask yourself: do I actually understand this? You try to explain it with what you already know. If you can't, you reach for connections, analogies, new frameworks. You're not consuming information — you're actively rebuilding your mental model.

That difference is the essential gap between next-token prediction and MTP.

For a large enough model, predicting the next token quickly becomes too easy. It can cheat by relying on local statistical shortcuts — "given these two words, the third is probably X" — without truly understanding the sentence. MTP changes the game: predict t+1, t+2, t+3, t+4 simultaneously. That's like being forced, after reading a paragraph, to anticipate not just the next sentence but where the whole section is going. You have no choice but to understand the logical structure beneath the surface.

**The mechanical trace:** The paper mentions that MTP encourages the formation of induction heads — attention circuits inside the Transformer that recognize repeated patterns, draw analogies, and align long-range context. Once a model develops induction heads, its few-shot learning ability makes a sudden jump. MTP, by raising the difficulty, forces these deeper inductive structures to emerge.

**One caveat worth emphasizing:** this difficulty doesn't break the model, because MTP is designed to be "just barely reachable," not impossible. The loss on distant heads is much higher than on nearby heads, but it's still slowly declining — not oscillating or exploding. That design choice leads to the next thing I find interesting.


---

### Insight 2: 70% Familiar, 30% Unknown — Letting Difficulty Grow with Capability

**The problem:** Every model is trained from start to finish on the same fixed objective — always predict 1 token, or always predict 4. But a model at the start of training is a "first-grader." By the end, it's a "PhD student." Assigning the same difficulty of homework to both states is clearly not the best strategy for either.

A good teacher doesn't hand first-graders a philosophy textbook, and doesn't make PhD students do basic arithmetic drills. The best way to keep a student growing is to keep the difficulty just slightly ahead of their current ability — always within reach if they stretch, but never out of reach. Too easy and the brain coasts, learning only surface patterns. Too hard and it gives up, learning nothing — or learning the wrong things. The sweet spot is roughly 70% familiar, 30% unknown. That's where capability keeps emerging.

**My proposed training schedule:**
- Start weak → predict 1 token, build a stable foundation
- After several rounds of improvement → move to 2–3 tokens
- Improve further → 4 tokens plus simple reasoning
- Keep scaling difficulty as the model gets stronger → add logic, causal structure, and beyond

The goal throughout: task difficulty tracks actual model capability, not a fixed value set from day one.

One of the things I enjoy most about reading great papers is how the researchers' thinking triggers ideas of my own — and then I can't help but go check if anyone is working in that direction. I'm often pleasantly surprised to find that they are. This was one of those times: ACL 2025 published an almost identical idea independently, under the name "Pre-Training Curriculum for Multi-Token Prediction" — which made me both happy (the intuition was right) and a little wistful (it's already been done). They showed that a forward curriculum, gradually transitioning from next-token prediction to MTP, works better — especially for small models.

**But I think there's a more specific gap:** instead of adjusting the difficulty of training data, what if we adjusted the number of MTP heads itself? Based on the model's current loss curve, dynamically upgrade from 2 heads to 3, then to 4, mid-training. Existing curriculum work focuses on data. This direction focuses on the dynamic evolution of the target architecture itself. These are different problems.

**How to tell if difficulty is "just right":** The paper gives you the signal. Healthy state means distant heads have high but slowly declining loss. Overloaded state means loss gets stuck, oscillates, or explodes. In practice, stepped loss weights offer a static version of this control — main head 1.0, MTP-1 head 0.6, MTP-2 head 0.3, MTP-3 head 0.1 — so the model isn't overwhelmed from the start. That's a static staircase. The real direction, I think, is making the staircase itself adjust dynamically throughout training.


---

### Insight 3: The Walking Gait Analyzer — Why Feedback Granularity Determines Learning Speed

Even within the MTP framework, each head still predicts a one-hot label — the model guesses a specific token, and it's either right or wrong. Predicting "cat" when the answer is "dog" and predicting "car" when the answer is "dog" receive the exact same penalty. The distance between errors is invisible.

**Analogy:** Imagine someone with poor walking posture. They walk ten thousand steps a day, but nobody tells them what's wrong. At the end of the walk, a coach shouts "you walked incorrectly." But they don't know if the issue is their center of gravity drifting left, their stride being too long, or their foot angle. They can only adjust randomly. That's one-hot training. The model knows it got the answer wrong, but not how wrong — not that "cat" is closer to "dog" than "car" is. It has to grope its way through massive amounts of data.

**A better signal:** Compare that to a real-time gait analyzer that says: "center of gravity 3 degrees left, stride 10cm too long, foot angle correct." Every step moves you toward the right direction. That's the idea behind embedding space loss: instead of predicting the next token directly, predict the embedding vector of that token. Guessing "cat" and guessing "car" now produce different penalties, because "cat" is geometrically closer to "dog."

**What I find interesting:** this pattern shows up in three seemingly unrelated techniques:
- **MTP's embedding space loss** — predict the token's embedding vector, not the token itself
- **Hinton's knowledge distillation (2015)** — student learns from teacher's soft probability outputs
- **Label smoothing** — distribute a small probability across all classes

Different forms, same underlying idea: don't just teach the model what's right, teach it how far wrong answers are from the right one.

Hinton's 2019 work found an interesting counterexample: if the teacher model was trained with label smoothing, it actually becomes a worse distillation teacher. The reason makes the logic clear — label smoothing causes same-class features to cluster too tightly, erasing subtle similarity differences between examples. And those subtle differences are exactly the "dark knowledge" that distillation needs to transfer. **The finer the feedback, the richer the structure the model learns.**

**The deeper problem:** knowing "how wrong you are" (distance) and knowing "why you're wrong that way" (cause) are two completely different things. Even if you tell someone their walking is off by 3 degrees, they still don't know whether it's a muscle problem, a habit, or their shoes. The loop that actually leads to fast improvement isn't endless repetition — it's:

1. Practice
2. Notice the gap
3. Understand why the gap appeared
4. Figure out how to fix it
5. Form a method, then enter the next round

The key isn't the number of iterations. It's that each iteration contains one act of structured reflection. Without that, more data just means more random walking.

**Mapping this to model training:**
- **Bottom level** — knowing right from wrong: "your answer is incorrect." Corresponds to one-hot loss. Requires enormous data just to find a direction.
- **Middle level** — knowing how wrong: "cat is closer to dog than car is." Corresponds to embedding space loss and knowledge distillation. Gives directional information, but still no causal explanation.
- **Top level** — knowing where reasoning went wrong, step by step. A direction I find fascinating, though I haven't studied it deeply enough yet.

The MTP paper's proposed future work moves toward the middle level. Whether the top level — where auxiliary loss includes step-level feedback signals, not just prediction distance — could be incorporated into MTP training is something I want to understand better.


---

### Insight 4: Intent Prediction — Moving the Battle Against Hallucination to the "Why"

MTP's four heads all predict highly correlated "next words" — t+1, t+2, t+3, t+4 — which are fundamentally lexical and statistical tasks. This made me wonder: could we add an auxiliary task at a completely different level?

**From my own experience working with LLMs:** the most effective way to avoid hallucinations isn't to keep correcting specific outputs. It's to ask a question before the model gives its answer — "why are you responding this way? what problem is this response solving?" If its intent matches mine, we can continue even if the wording is a bit off. But if the intent has already drifted, no matter how fluent the output is, everything that follows will go wrong — and the more fluent it is, the more misleading.

**My training idea:** what if the model, while predicting tokens, also predicted "what question is this sentence answering" or "what is the speaker's intent"? Not the next word — but the logical purpose behind the entire generation.

- MTP addresses **local fluency**: deeper token-level dependencies
- Intent prediction would address **global consistency**: maintaining logical direction at the paragraph level
- The first treats "words that don't convey the meaning." The second treats "heading in the completely wrong direction." These aren't in conflict — you can do both at once.

What makes me think this direction is worth exploring: the research community is already doing something similar at inference time. Chain-of-Intent, Self-Ask, and React all get the model to explicitly output intent or a reasoning path before generating an answer. But these are inference-time interventions, not structural changes at training time. Turning intent prediction into an auxiliary training loss — I haven't found published work on this, which suggests it may be a real gap.

**A lightweight validation path:**
1. Use a capable LLM to auto-generate intent annotations for existing corpora — "what is this passage doing? explaining a concept, giving advice, or describing a fact?"
2. Attach a lightweight intent classification head on top of the model's hidden states
3. Train jointly: main task (predict next token) + auxiliary task (predict current intent)
4. If it works, we should see hallucination rates fall and logical consistency improve on multi-step reasoning tasks


---

### Insight 5: Information Flow vs. Information Visibility — Why Pause Tokens Can't Replace MTP

There's an experiment in the paper I initially thought was testing MTP's robustness. On closer reading, it was doing something more important.

**The experiment:** The authors swapped the attention mask from Causal (look only backward) to Prefix (bidirectional prefix + causal suffix) to Full (global bidirectional), then compared single-token models and MTP models on reasoning tasks. The result: even expanding the single-token model's view to fully bidirectional, it still couldn't match a MTP model using only causal masking.

This experiment pre-answers a plausible objection: that MTP's stronger reasoning is just because it indirectly gives the model access to more context — that the training objective itself isn't the real source of the gain. The experiment closes that door completely.

**The clearest way to understand this:** consider two completely different kinds of challenges. Giving students an open-book exam solves "can they see the information." Training students from the beginning to break problems into steps, track intermediate reasoning, and close logical loops solves "can they remember, carry forward, and reason through information step by step." Even with a fully open book, the student trained in structured reasoning will clearly outperform on long, complex problems. These are not the same dimension.

**Pause Tokens vs. MTP:** This also cleared up something I'd always found slightly mysterious: Pause Tokens. Inserting blank tokens during generation to give the model "thinking time" — like the brain's default mode network going quiet so it can reorganize information in the background. I genuinely like this idea. But Pause Tokens solve the same class of problem as expanding the attention window. Both are "give a bit more time and space at runtime" optimizations. They're patches added in the moment.

MTP is doing something else. It restructures the model's information flow at training time — forcing the hidden state, as it's being written, to already carry preparation for predicting the next step, the step after, and the step after that. That internal flow structure can't be installed after the fact with a Pause Token.

**For engineering practice**, I think this distinction matters a lot:
- **Runtime optimizations** (Pause Tokens, Prefix Attention, KV cache management) — make the model's existing capabilities more fully expressed. Engineering work.
- **Training-time restructuring** (MTP, CoT fine-tuning) — fundamentally reshapes what capabilities the model has in the first place. Research work.

What makes MTP unusual is that it does both at once: restructuring capabilities at training time, then at inference time naturally serving as a built-in draft for speculative decoding. Two layers of value, one model.


---

### Insight 6: Progressive Distillation + Dissection — From Big-Lab Monopoly to Open Innovation

MTP raises the ceiling on reasoning capability. But there's a very practical problem: the number of teams that can train an MTP model from scratch fits on one hand. For most people, that path simply doesn't exist.

I think there's a way around it.

The large models that have already been trained — their compute is a sunk cost. The world knowledge, reasoning ability, and inductive structure inside them already exist. I don't need to regenerate those capabilities. I just need to extract them.

**Why direct pruning doesn't work:** pruning a large model directly is tempting, but it almost never succeeds. Knowledge in a large model is diffuse — different layers and different attention heads all participate together in a single capability. There's no individual weight solely responsible for "reasoning" or "world knowledge." Cut the wrong head and reasoning collapses. It's like randomly disassembling a precision machine and expecting the remaining parts to still run.

**The better approach — progressive distillation:** not one big step from large to small, but large → medium → small, one stage at a time. At each step, the small model isn't receiving a "truncated large model." It's learning the large model's way of thinking — its prediction distribution, its attention patterns, its inductive structure. The result isn't a diminished version; it's a genuine smaller reproduction of how the large model reasons.

Once distilled small enough, the model reaches a "clean" state: no redundant weights, no scattered knowledge, only core language structure and reasoning ability. At that point it becomes analyzable — you can use SAE (Sparse Autoencoder) to dissect its internal feature representations and understand which circuits correspond to which capabilities. Then, feeding it a small amount of high-quality, high-difficulty data works extremely well. The underlying capability is already there; it only needs to deepen and refine, not start from scratch.

**My proposed two-stage pipeline:**
- **Stage 1 (big companies / resource teams):** large model → progressive distillation → small clean model → open-source to the community
- **Stage 2 (individuals / small teams):** small model → SAE dissection → identify key circuits → fine-tune with high-quality data → a personally capable model

This is completely different from standard fine-tuning, which patches a large model full of redundancy. This path extracts the intelligent skeleton first, then evolves it further. Fine-tuning is dressing up a heavy model. This is distilling the structure to its essence, then cultivating it again.

**What excites me most:** this breaks the resource barrier in large model research. It transforms "train a capable model" from "we need thousands of GPUs" into "we need a few GPUs, a high-quality dataset, and enough patience."

If the large model trained with MTP is the starting point of this distillation path, the implications are even larger — because MTP gives that large model deeper, more structured reasoning. The small model that inherits from it carries that depth of reasoning, not just surface language ability. From this angle, MTP isn't just a training optimization for a single model. It's the seed paradigm for an entire next generation of lightweight, high-reasoning models.


---

> Published on Medium: https://medium.com/p/3859ab7e4037?postPublishedType=initial

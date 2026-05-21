# Rethinking How LLMs Remember: From KV Cache to Human-Like Learning — DeepSeek-V4 Technical Report (DeepSeek-AI, 2026)

- Paper: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf

> What follows is not a summary of the paper. These are my own thoughts, analogies, and brainstorm ideas.
> All insights are exploratory and unverified — they represent directions I find worth exploring, not claims backed by experiments or evidence.
>
> **One thing I noticed:** DeepSeek-V4 is full of surprising design choices. Every time something seemed counterintuitive to me, digging deeper revealed a deliberate tradeoff — a gain worth the cost. The right question when something surprises you is always: "what did they optimize for by doing it this way?"

---

## I. Ideas That Emerged While Reading

### 1. What If the Model Learned to Manage Its Own KV Cache?

**Core idea:**
> Today, all KV cache management is rule-driven — evict by recency (SWA), or retain by attention score (H2O).
> But these are all rules designed by humans, applied to a frozen model. The model itself never participates in the decision.
> **What if we trained the model to learn: "this conversation turn is useless, drop it; that one is critical, keep it"?**

**Capabilities imagined:**

> Existing approaches — StreamingLLM, H2O, SnapKV — all rely on human-designed heuristics applied after training. The model itself never participates.

1. **Task-aware dropping** — the model knows what question it's answering, so it can judge which KV entries matter *right now*.

2. **Zero-footprint dropping** — filler turns, repetitive prompts, meaningless small talk → entire turn deleted, KV completely gone. 

3. **Merge (consolidate)** — not just discard, but consolidating semantically similar, redundant multi-turn exchanges into one "summary KV." Like human memory: not storing frame-by-frame video, but storing "the most important thing that happened today." 

4. **Paradigm distillation** — after seeing many examples of the same pattern, distill into one general principle and drop the individual examples. Like a student who no longer needs flashcards after the rule is internalized. Ten instances → one abstract "I know this pattern."

5. **Story threading** — instead of storing every fact separately, link context into a coherent narrative thread. Models have strong reasoning ability — they don't need every detail in KV if they can reconstruct it from a compact "story spine" + inference. Store the thread, reason out the rest.


**The one-sentence core of this insight:**
> Merge is the sharpest part.
> Not "which tokens should stay," but "which tokens can be replaced by a smaller vector" —
> this is the paradigm shift from KV eviction to KV consolidation.

---

### 2. The Missing Third Side of the Triangle — Why Models Can't Grow Like Humans

**Core observation:**
> MTP looks forward (predicts future tokens);
> CSA looks backward (efficiently retrieves past KV);
> Together they determine the current output — like a human decision that considers both past experience and future goals.
>
> But the triangle has a missing third side.

```
        FUTURE
          ↑
    MTP: predict future tokens

           NOW

CSA: efficiently retrieve past KV       ???: compress past into ability
(past exists as "raw memory")           (past internalized as skill, raw record no longer needed)
```

**How humans achieve the third side:**
> A doctor doesn't re-read all patient files to make a diagnosis —
> past cases have crystallized into "clinical intuition," living in the doctor herself; the cases can be "forgotten."
>
> A chess grandmaster doesn't re-analyze the board from scratch —
> tens of thousands of games have compressed into "pattern recognition ability"; the game records no longer need to be kept.
>
> **Human memory path: experience → forms ability → ability persists, original experience can be discarded**

**The model's problem**
```
Humans:
  Experience  →  forms Ability  →  Ability persists (no need to re-access original experience)

Models:
  Context  →  uses Context  →  close context  →  reset to initial level
  ↑ Every request starts from the same baseline, no matter how many times done before
```

> The model's context window is "working memory," but there's no mechanism to convert
> what's learned inside working memory into "long-term ability" — no Memory Consolidation mechanism.

**Brainstorm: Auto-Learn — let the model accumulate capability**
> If after each session, the model could distill "what did I learn this session"
> into new capability, baked into its own weights —
> then each new request would start from a higher baseline, not always the same initial level.

```
Ideal model growth path:
  Session 1: [level 0] + context → learns something → bakes into weights → [level 1]
  Session 2: [level 1] + context → learns more → bakes in → [level 2]
  Session N: [level N-1] → every request stands on all prior sessions' shoulders

Reality:
  Session 1: [level 0] → close → [level 0]
  Session 2: [level 0] → close → [level 0]
  Forever at the same starting point
```

**Deep connection to MTP + CSA:**
> MTP and CSA both solve efficiency problems *within* a single session.
> The insight: **continuous growth *between* sessions** is the real gap that would make models approach human-level development.
>
> With the triangle completed:
> - MTP (looks forward) + CSA (looks backward at KV) + Auto-Learn (converts past into capability)
> = a system that can plan, recall, and grow
---

### 3. What Caught My Eye: The KV Cache Deliberately Sent to Disk

**What surprised me:**
> My assumption going in: disk is emergency overflow — GPU HBM runs out, you reluctantly spill to disk.
> DeepSeek-V4 does the opposite. Compressed KV (CSA/HCA) is **proactively written to disk during prefill** — planned from the start, not triggered by memory pressure.
> The disk isn't a fallback. It's a designed memory tier.

**What this enables:**
> When many requests share the same prefix — same system prompt, same document, same instructions — the prefix is computed once, archived to disk, and loaded on demand by every subsequent request.
> **Prefill cost paid once. Every future request gets a cache hit.**
>
> The system doesn't just tolerate the disk — it plans around it. Three tiers working together: GPU HBM (active SWA window) → CPU DRAM (warm buffer) → NVMe disk (compressed prefix archive).

**A confusion worth sharing: if CSA covers all tokens, why does every layer also have SWA?**

> My first reading: SWA is a staging area for tokens that haven't formed a complete CSA block yet.
> When 4 tokens accumulate (block size m=4), they compress into a CSA block and SWA moves on. Right?
>
> Wrong. SWA window = 128 tokens (Pro) — far more than the 3 incomplete tail tokens. And the bigger correction: **tokens live in both simultaneously.**

```
Token born → enters SWA window (uncompressed, full precision)
   ↓
4 tokens accumulate → also compressed into CSA block (long-range)
   (BOTH representations exist at the same time)
   ↓
Token slides past n_win = 128 boundary
   → evicted from SWA → now lives only in CSA block
```

> SWA isn't a staging area. It's a **permanent local window** — always keeping the most recent 128 tokens at full resolution, regardless of whether those tokens are also in CSA blocks.
>
> Why? Because CSA compression is lossy: m=4 tokens → 1 vector → fine-grained local detail is gone.
> For the tokens you're about to use next, that loss matters most. So they stay uncompressed in SWA — even though they're also in CSA. Tiny redundant cost. Full local precision recovered.

**The on-disk implication of this:**
> Both SWA and CSA KV go to disk for the shared prefix — which is why SWA on-disk storage is 8× larger than CSA.
>
> CSA on disk: N/4 blocks × compressed dim (small per entry)
> SWA on disk: N tokens × full KV dim (uncompressed, full size)
>
> At first glance it looks wasteful. But 128 tokens against 1M context is tiny — the redundant storage cost is minimal, and it buys back full local precision completely.

**The working memory view:**
> KV cache is the model's working memory — how large it is determines how complex a task the model can handle.
> By deliberately designing disk as a storage tier, the model's effective working memory extends far beyond what fits in GPU.
> The architecture isn't fighting hardware limits. It's building around them by design.

---

### 4. Vertical Growth vs. Horizontal Reuse — The Unsolved Problem of On-Disk KV

**What I noticed:**
> DeepSeek-V4's on-disk KV only solves the "horizontal problem" — same prefix, shared across many requests.
> But real agent scenarios are "vertical growth" — one session, thousands of turns, growing longer and longer.

**The contrast:**

```
Horizontal (paper solved):
  Request 1: [system prompt | doc] + "summarize"
  Request 2: [system prompt | doc] + "translate"
  Request 3: [system prompt | doc] + "find errors"
            └──── shared prefix ────┘  └─ unique ─┘
  → prefix computed once, reused by many requests, stored on disk

Vertical (Open Problem):
  Turn 1:   [prompt] → 500 tokens generated
  Turn 2:   [prompt + turn1] → 500 more
  Turn 50:  25K tokens, HBM OK
  Turn 500: 250K tokens, HBM under pressure
  Turn 2000: 1M tokens → old turns must be offloaded
```

**What I kept imagining:**
> What if old generated turns were progressively compressed into CSA blocks and offloaded to disk — the same way the shared prefix is handled?
> Over time, the boundary between "what I generated" and "what counts as shared prefix" would blur.
> The session's own history becomes its own growing prefix.
> I don't think DeepSeek-V4 does this — but it feels like the natural next step: as the session grows, old turns get compressed and pushed to disk while inference keeps running, so the model can still attend back to turn 1 even at turn 2000.

---

## II. Analogies

> When I hit something hard to understand, I reach for a life analogy — something concrete I already know. The following are the ones that helped me most while working through this paper. If you're reading the paper and getting stuck, these might help.

### 1. CSA = Library Card Catalog + Book Retrieval

**The analogy:**
> K^IComp (lightweight indexer keys) = the card catalog — one small card per book, sitting outside the stacks.
> C^Comp (compressed KV blocks) = the actual books — stored in the warehouse, heavy, can't all be moved at once.

**How it actually works:**
1. Check the card catalog (K^IComp stored entirely in HBM) → cheap, fast, 500K cards all in GPU memory
2. Find the most relevant top-k books (sparse selection: Pro picks 1024, Flash picks 512)
3. Fetch only those books (load C^Comp from disk) → run the actual attention

**Why this analogy works:**
> It explains why Lightning Indexer needs two separate KV systems —
> the catalog (lightweight) and the books (heavyweight) are designed for fundamentally different purposes.

**CSA's full three-axis design:**
```
① Compression (reduces KV memory)
   Every m=4 tokens → 1 block (C^Comp)
   KV: n × c → (n/4) × c  (4× fewer entries)

② Sparse selection (reduces attention compute)  ← this is what the card catalog covers
   K^IComp stored fully in HBM, cheap scoring
   Q · K^IComp → scores → select Top-k blocks
   Compute: O(n²) → O(n · k)

③ Shared KV MQA (one vector, two jobs)
   Standard: K and V are separate (each specialized)
   CSA: K = V = C^Comp — one vector learns both "how to match" AND "how to retrieve content"
   Cost: lose K/V specialization  |  Gain: KV storage halved
```

**Complexity improvement:**
```
Phase              Complexity     1M context example
Prefill            O(n²)          1M × 1M = 1T ops (catastrophic)
Each decode step   O(n)           1M dot products (still expensive)

After CSA:
Prefill            O(n × k)       k = top-k blocks, far smaller than n
Each decode step   O(k)           only top-k blocks, not full n
```

---

### 2. CSA + HCA + SWA = Three Layers of Painting Composition

I like painting, so when I hit this part of the architecture I naturally reached for that analogy.

When you compose a scene, the background can be impressionistic — broad strokes, suggested mountains, no need to render every leaf. The midground has more detail, but only where it matters. The foreground — whatever your eye rests on — is sharp, fully resolved, every brushstroke present.

That's exactly what CSA, HCA, and SWA are doing. HCA is the far background: 128:1 extreme compression, covering the full 1M-token context in the lowest resolution. CSA is the midground: 4:1 compression plus sparse selection, drawing only the most relevant blocks from history. SWA is the foreground: the most recent 128 tokens (both Pro and Flash), fully uncompressed, full precision.

The reason this feels right to me: in painting, "near things matter most" isn't a stylistic choice — it's a natural law of how eyes work. The architecture follows the same principle. SWA exists in every layer, uncompressed everywhere, because local precision matters everywhere — not just in some layers.

The actual layer structure confirms this. Every single layer has SWA built in as a branch — it's not optional or selective:

```
DeepSeek-V4-Pro (61 layers):
  Layer 1-2:   HCA only
  Layer 3-61:  interleaved CSA / HCA
                └── every layer has SWA branch (n_win = 128 tokens)

DeepSeek-V4-Flash (43 layers):
  Layer 1-2:   SWA only
  Layer 3-43:  interleaved CSA / HCA
                └── every layer has SWA branch (n_win = 128 tokens)

What "interleaved" means:
  Layer 3:  CSA  (+ SWA branch)
  Layer 4:  HCA  (+ SWA branch)
  Layer 5:  CSA  (+ SWA branch)
  ...
```

The foreground is always there. No matter which layer, no matter whether it's doing long-range CSA or extreme-compression HCA — the local window is always preserved alongside it.

---

### 3. KV Storage Tiers = Three Levels of Human Memory

```
GPU HBM (working memory)  → what you're doing right now, instantly available (SWA window + current decode KV)
CPU DRAM (short-term)     → what you just finished, retrievable in seconds (warm compressed KV copy)
NVMe SSD (long-term)      → what happened long ago, requires "looking it up" (full compressed CSA/HCA KV)
```

> DeepSeek-V4's innovation: transforming SSD from "forced fallback" into an **actively designed** storage tier.
> Not waiting for HBM to overflow before swapping — proactively archiving after prefill completes.
>
> I wonder if this is part of why storage hardware stocks (Micron, Samsung, Seagate) have been
> getting attention lately — if NVMe becomes a standard inference tier, not just a backup,
> the demand story for storage changes completely.

---

### 4. Base Model = Bare Concrete Shell; Instruct Model = Move-In Ready Apartment

> Base model = freshly poured concrete shell — solid foundation, straight load-bearing walls, but no plumbing or power, uninhabitable.
> Instruct model = fully finished apartment — plumbing connected, kitchen and bathrooms complete, move in immediately.

**Why DeepSeek published both — and why the base model matters (from my point of view):**
> The instruct model is for users. The base model is for the research community.
>
> Publishing the base model is like handing the bare concrete shell to every interior designer in the world — each team decorates it their own way. A medical team fine-tunes it on clinical data. A coding company runs their own RLHF. A university lab experiments with new post-training methods. Everyone skips the foundation-building and goes straight to the part only they can do.
>
> Training a base model at this scale costs enormous compute — most teams can't do it. By releasing the base model, DeepSeek gives the whole research community access to the foundation.
>
> That's what makes it a real contribution, beyond just shipping a product. The instruct model closes a release. The base model opens a hundred research directions.

**The four models:**
```
V4-Pro Base       → 1.6T params, 49B activated — foundation for researchers to fine-tune
V4-Pro Instruct   → same + SFT + RL (GRPO) + OPD — agent-focused, ready for end users
V4-Flash Base     → 284B params, 13B activated — lightweight foundation
V4-Flash Instruct → same + post-training — speed/cost optimized
```

---

### 5. Why Long-Horizon Matters

> Real-world tasks aren't single questions — they're projects. Writing a research paper, building a software feature, running an experiment. These take many steps, each depending on the last.
>
> Long-horizon is the ability to hold a goal, execute a plan, recover when something goes wrong, and still arrive at the right answer many steps later. That's what makes AI genuinely useful for the hard things — not just answering questions, but actually doing things that take real time.

**The analogy: Finishing a Thick Book vs. Completing a 3-Month Project**
> Long-context = a person can read and retain a very thick book (memory capacity)
> Long-horizon = a person can execute a project that runs for three months (planning ability)
>
> Long-context is the foundation. But memory capacity ≠ planning ability.
> The more steps involved, the easier it is to forget the original goal (goal drift); one wrong step in the middle causes everything downstream to fail.

> This is exactly why DeepSeek-V4's post-training (GRPO + OPD) targets agent capability — long-horizon isn't just a nice property, it's what the whole post-training pipeline is building toward.

**What is an agent?**
```
LLM      = the brain (understand + generate) — DeepSeek V4 itself
Agent    = brain + tools + memory + action — LLM as core, plus planning
Workflow = multiple agents in a pipeline — a team where each agent has a role

DeepSeek V4 = an LLM trained to excel at agentic behavior
(not an agent application like Claude Code — but can serve as the brain inside one)
```


---

The end.

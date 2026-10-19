---
title: "Keeping the Draft Walk on Chip: One Launch for DSpark's Markov Head"
description: "DSpark drafts a whole block in parallel, then corrects it with a serial Markov walk that re-reads a vocabulary-sized matrix at every step. We run the whole walk as one persistent kernel with the matrix held on chip: 8-12x faster walks on H100, up to +12% end to end, identical outputs, plus bf16 tensor-parallel and B200 versions."
authors: "Xinwei Qiang, Ziyi Xu, Zaifeng Pan, and Yufei Ding"
date: "October 19, 2026"
tags:
  - Speculative Decoding
  - CUDA
  - GPU Kernels
  - LLM Inference
---

**TL;DR:** DSpark drafts a block of γ tokens in one parallel pass, then fixes it up with a small *serial* walk. Every step of that walk multiplies by a matrix as wide as the vocabulary. We run the whole walk as one persistent kernel that keeps the matrix on chip.

- **H100, the walk alone:** 331 → 37 µs at γ = 7 and 741 → 62 µs at γ = 16 (bs = 1).
- **H100, end to end in SGLang:** +5.3–5.9% per round at γ = 7 and +10.8–12.1% at γ = 16, at concurrency 1–16. Outputs are identical, 605/605.
- **Tensor parallelism:** splitting the matrix over two GPUs lets it stay on chip in bf16, with no quantization. That wins from batch 8 up.
- **B200:** a third layer of on-chip memory, TMEM, holds the whole bf16 matrix: 4.5× faster at γ = 16. For a vocabulary too big to fit, we keep 62% on chip and stream the rest: 4.4× faster.
- **Other serial heads** (Domino, DeLS, DSpark's gated and RNN heads) gain 6.6–9.0× over their exact reference paths, from the same kernel template. Domino gains +16.8% per round end to end.

Code: [github.com/tie-pilot-qxw/dspark-markov-walk](https://github.com/tie-pilot-qxw/dspark-markov-walk) (public since October 4) · Upstream PR: [sgl-project/sglang#42855](https://github.com/sgl-project/sglang/pull/42855) (open since October 6)

---

## The serial part of a parallel drafter

A block drafter such as DSpark guesses a whole block of tokens in one forward pass. That pass is parallel, so it is cheap. The catch is that every position is guessed blind: position 3 does not know what positions 1 and 2 turned out to be.

DSpark wins part of that back with a small Markov head. After the parallel pass, it walks the block from left to right. At each step it adds a bias that depends on the token it just picked:

```
prev = anchor
for k in range(γ):
    u       = W1[prev]                 # [256]
    logit_k = bf16(base_k + W2 @ u)    # [V]
    tok_k   = argmax(logit_k)          # or a sample of softmax(logit_k / T)
    prev    = tok_k
```

Each step needs the token from the step before, so the steps cannot run in parallel.

SGLang's default walk runs them as γ separate steps. Each step is a full-vocabulary GEMV, then an add, an argmax and a sampler. W2 is V × 256 in bf16: 78 MB at Qwen3's 151,936-token vocabulary, and 127 MB at 248,320. So every round reads the same tens of megabytes from HBM γ times, just to pick one token per step.

That adds up. On an H100 at bs = 1, the walk costs **331 µs per round at γ = 7 and 741 µs at γ = 16**. That is 5.8% and 12% of a whole Qwen3-4B speculative-decoding round (5.72 ms and 6.06 ms). The big target model verifies the draft in parallel. This little serial tail is what is left.

## The idea: make the walk resident

Two facts about the walk point to the design:

- **What changes between steps is tiny:** one token id.
- **What each step reads never changes:** the same W2, every step, every round.

Persistent RNN kernels used exactly this shape ten years ago. They kept the recurrent weights on chip and kept one kernel alive across all time steps [(Diamos et al., 2016)](https://proceedings.mlr.press/v48/diamos16.html). We apply the same idea to the draft walk.

First, the matrix has to fit. An H100 has 65.4 MB of on-chip storage across its 132 SMs (registers plus shared memory). bf16 W2 is 78 MB, so it does not fit. An int8 W2 with one fp32 scale per row is 38.9 MB, and that does.

So the kernel works like this:

- **One CTA per SM, one launch per round.** Each CTA owns a slice of vocabulary rows, 1,152 rows at V = 151,936. It loads its slice of int8 W2 into registers and shared memory once. Then it runs all γ steps without leaving the kernel. W2 is read from memory once per round instead of γ times.
- **Exact integer math.** u = W1[prev] is stored as two int8 planes, u ≈ s_hi·q_hi + s_lo·q_lo with s_lo = s_hi/256. Both dot products run as exact int32 sums on the int8 tensor cores, so nothing is dequantized inside the loop.
- **One global argmax per step, with no grid barrier.**
  - Each CTA reduces its rows to one 64-bit key: the order-preserving bits of the fp32 logit, then the inverted row index, so ties go to the smallest token id.
  - Every CTA posts its key to one 16-byte `{key, count}` pair, with an atomic max on the key and an atomic add on the count. Then it polls the pair until the count says every CTA has arrived.
  - There is no `grid.sync`. There is also no memset between CUDA-graph replays, because the per-round state alternates by round parity.
- **Sampling inside the kernel.**
  - Philox noise is keyed by (seed, row, step, round counter), so every graph replay draws fresh noise.
  - Small batches use per-row Gumbel-max. Large batches use an inverse CDF inside 8-row groups and Gumbel-max across groups.
  - Temperatures are read per request on the device, so one captured graph serves greedy, sampling and mixed batches.
- **Lossless verification.** The logits are rounded to bf16 once. The token is picked from exactly those values, and the same bits are written out for SGLang's verifier, which rebuilds the draft distribution from them. So the rejection sampler sees the distribution the draft was really drawn from.

Three kernels cover the batch sizes:

| kernel | batch | where W2 lives |
|---|---|---|
| `single` | bs = 1 | int8 `mma.sync` A fragments in registers and shared memory |
| `small_batch` | bs 2–4 | same layout; one MMA pass serves several requests |
| `wgmma` | bs 5–64, and every bs at V = 248,320 | W2 is the shared-memory operand and the request latents are the register operand; tiles that do not fit on chip stream from L2 through a TMA ring |

## Results on H100

All numbers in this section: H100 80GB SXM, CUDA 13.1.

### The walk alone

CUDA-graph replay, cold L2, µs per call. The baseline is SGLang's default walk.

| V | γ | bs | ours, greedy / T = 1 | default walk | speedup (greedy) |
|---|---|---|---|---|---|
| 151,936 | 7 | 1 | **36.8** / 37.9 | 331.4 | 9.0× |
| 151,936 | 7 | 16 | **70.5** / 90.1 | 569.6 | 8.1× |
| 151,936 | 7 | 64 | **115.7** / 171.5 | 988.9 | 8.5× |
| 151,936 | 16 | 1 | **61.7** / 65.6 | 740.7 | 12.0× |
| 151,936 | 16 | 64 | **265.5** / 389.4 | 2243.9 | 8.5× |
| 248,320 | 15 | 1 | **211.4** / 253.0 | 971.1 | 4.6× |
| 248,320 | 15 | 64 | **456.5** / 669.9 | 3296.7 | 7.2× |

At V = 248,320 even the int8 table (63.6 MB) no longer fits on chip. The `wgmma` kernel keeps as much as fits in shared memory and streams the rest from L2 at every step. That is why bs = 1 costs 211 µs there (γ = 15), against 62 µs on the Qwen3 vocabulary (γ = 16).

### End to end in SGLang

How we measured:

- **Workload:** Qwen3-4B target, 605 prompts from 9 datasets (gsm8k, math500, aime25, humaneval, mbpp, livecodebench, mt-bench, alpaca, arena-hard-v2), up to 2,048 new tokens.
- **Metric:** cost per round from the scheduler's decode log (`accept len / gen throughput`), p25, paired by batch size.
- **Protocol:** two interleaved passes per arm, with the order reversed in the second pass, and default server flags.

Per-round throughput gain:

| drafter | temp | conc 1 | conc 4 | conc 16 | bs ≈ 64 |
|---|---|---|---|---|---|
| `deepseek-ai/dspark_qwen3_4b_block7` (γ = 7) | 0 | **+5.5%** | +5.9% | +5.3% | +3.8% |
| `deepseek-ai/dspark_qwen3_4b_block7` (γ = 7) | 1 | **+5.2%** | +5.3% | +3.5% | +3.5% |
| `shreybirmiwal/Qwen3-4B-DSpark-b16` (γ = 16) | 0 | **+10.8%** | +12.1% | +11.2% | +6.1% |
| `shreybirmiwal/Qwen3-4B-DSpark-b16` (γ = 16) | 1 | **+10.5%** | +11.7% | +10.5% | +4.4% |

- **At bs = 1** a round goes from 5.721 to 5.424 ms at γ = 7, and from 6.059 to 5.467 ms at γ = 16.
- **A bigger target:** with Qwen3-14B and `deepseek-ai/dspark_qwen3_14b_block7`, the gain is +2.2% at concurrency 1. The kernel saves the same 0.29 ms per round as with the 4B target, but a 14B round takes 13.7 ms, so the share is smaller.

This is the general pattern: the kernel saves a fixed amount of time per round, so the percentage depends on how long the rest of the round takes. Longer blocks have longer walks and gain more. Bigger targets have longer rounds and gain less.

### Correctness

- **Outputs are identical.** With `--enable-deterministic-inference` at temperature 0, the generated text matches the default walk on **605/605** prompts for both drafters, at concurrency 1, 16 and 64.
- **Acceptance length is tied.** Ratio ours / default, with 95% prompt-cluster bootstrap intervals:
  - γ = 7: −0.09% [−0.36, +0.17] at temperature 0, and +0.11% [−0.39, +0.62] at temperature 1.
  - γ = 16: +0.09% [−0.34, +0.50] at temperature 0, and +0.47% [−0.20, +1.07] at temperature 1.
- **The drafts themselves can differ.** W2 is int8, with 0.70% relative quantization error, so near-ties can go the other way. That only changes how many draft tokens get accepted. The target still checks every token, so the output does not change.

## Tensor parallelism: bf16 with no quantization

int8 is what lets W2 fit on one H100. With two GPUs there is another option: split W2 by vocabulary. Each GPU then holds half of the bf16 table, 38.9 MB, which is the same size as the whole int8 table. So the walk can stay on chip with the head's own bf16 weights and fp32 accumulation.

The price is one more exchange. Each GPU finds the best token in its own half, and the two halves have to agree before the next step can start. We do this inside the kernel through peer-mapped mailboxes. The cross-GPU hop costs about 1.1–1.6 µs per step.

Round time at γ = 7, device clock, µs, greedy | T = 1:

| batch | int8, 1 GPU (best kernel) | bf16, 2 GPUs |
|---|---|---|
| 1 | **32.8** \| **33.5** | 52.0 \| 57.3 |
| 2 | **40.7** \| **47.5** | 54.3 \| 63.0 |
| 4 | **44.3** \| **54.4** | 55.6 \| 64.2 |
| 8 | 65.8 \| 84.8 | **58.9** \| **67.6** |
| 16 | 68.8 \| 86.0 | **65.7** \| **76.8** |
| 32 | 78.1 \| 92.8 | **72.6** \| **80.5** |
| 64 | 114.3 \| 144.4 | **87.2** \| **104.6** |

- **Small batches:** int8 on one GPU wins. At bs = 1 the hop costs ~8 µs per round, more than the bf16 version saves.
- **From batch 8 up:** bf16 on two GPUs wins, by 5–24% greedy and 11–28% at T = 1.

We have not wired the split version into SGLang:

- SGLang runs one process per TP rank, while the in-kernel exchange needs the ranks to see each other's memory, so serving would need CUDA-IPC mailboxes.
- Sampling would also need an all-gather of the corrected logits.
- We estimate the whole thing would be worth less than ~1% end to end.

Under SGLang TP > 1, the int8 kernels instead run the full walk on every rank, as SGLang's default walk does. With the reference hook in our repository, the `stepnivlk/Qwen3.6-27B-DSpark` drafter (γ = 15, V = 248,320) on a TP2 Qwen3.6-27B target gains **+5.4%** per round at bs = 1. The upstream PR keeps TP > 1 on the default walk for now. The bf16 kernels are in the repository under `experimental/bf16_tp/`.

## B200: the bf16 table fits on chip

Blackwell adds a third kind of on-chip memory. Each SM gets 256 KB of tensor memory (TMEM) next to its registers and shared memory, and only the new `tcgen05` tensor-core instructions can read it. Across 148 SMs that gives:

| | H100 SXM | B200 |
|---|---|---|
| SMs | 132 | 148 |
| on chip (registers + shared memory + TMEM) | 65.4 MB | **112.1 MB** |
| L2 | 50 MB | 126.5 MiB |
| tensor-core instructions | `mma.sync`, `wgmma` | `mma.sync`, `tcgen05` (no `wgmma`) |

The bf16 W2 for Qwen3's vocabulary is 77.8 MB. That does not fit on an H100, but it does fit on a B200. So on B200 we keep the whole table in bf16 with no quantization:

- 147 CTAs × 1,040 rows;
- 46% of the rows sit in TMEM, 22% in shared memory, and 32% in registers;
- `tcgen05` reads the TMEM part, and `mma.sync` handles the rest.

It uses the same rounding points as SGLang's walk: fp32 accumulation, the bias rounded to bf16, then the sum rounded to bf16. On our test data the drafts and corrected logits matched the default walk bit for bit. The accumulation order differs, though, so that is not a guarantee.

**A caveat about the numbers.** We ran on rented B200s (Modal). We could not lock clocks, and our kernel and the baseline ran in different containers rather than interleaved. Read every B200 ratio as ±10%.

### V = 151,936: the whole bf16 table on chip

Walk alone, µs per call, against the fastest SGLang walk for each cell:

| γ | bs | ours (bf16), greedy / T = 1 | fastest default walk, greedy / T = 1 | speedup |
|---|---|---|---|---|
| 7 | 1 | **42.2** / 43.6 | 152.0 / 152.5 | 3.6× / 3.5× |
| 16 | 1 | **72.2** / 75.3 | 324.6 / 325.1 | 4.5× / 4.3× |
| 16 | 4 | **97.1** / 109.1 | 460.9 / 459.8 | 4.7× / 4.2× |
| 16 | 8 | **114.9** / 144.8 | 543.5 / 581.2 | 4.7× / 4.0× |

The ratios are smaller than on H100. That is because the baseline gets faster too: the whole bf16 table fits in B200's L2, so the default walk drops from 44.0 µs per step on H100 to 19.2 µs on B200. Our int8 kernel, recompiled for B200, barely moves: 2.73 µs per step on H100, 2.82 on B200.

Keeping the table on chip still matters. A probe that reads the whole bf16 table from L2 every step, inside a single launch, needs 6.76 µs per step. The resident kernel needs 3.34 µs, about 2× faster.

Where one step goes, at bs = 1:

- the GEMV takes 1.43 µs;
- the cross-SM argmax exchange takes about 1.3 µs. Even with only 8 CTAs it takes 0.97 µs, so the cost is coherence between B200's two dies, not contention. Every alternative we tried was slower: `grid.sync` 2.17 µs, mailboxes 1.67–2.10 µs.

Batching is almost free for the tensor cores, because request n is column n of the MMA's N = 8 operand. What grows with batch size is everything around the GEMV.

### V = 248,320: 62% on chip, the rest streamed

At 248,320 tokens, the bf16 table is 127 MB, and even B200 can only keep 78.8 MB (62%) on chip. Every step, the other 48.5 MB has to come from L2:

- One thread of warp 0 copies the streamed rows with TMA into a 64 KB shared-memory ring, 16 KB at a time.
- The same thread issues `tcgen05` MMAs straight from shared memory.
- Warps 1–7 work through the resident tiles in parallel.
- W2 does not depend on the token, so the ring keeps refilling across steps. The next step's first stages are already in flight during the current step's argmax.

bs = 1, µs per call:

| γ | ours, greedy / T = 1 | fastest default walk, greedy / T = 1 | speedup |
|---|---|---|---|
| 1 | **27.5** / 28.8 | 37.2 / 45.0 | 1.35× / 1.56× |
| 15 | **116.2** / 119.3 | 510.1 / 527.3 | **4.4×** / 4.4× |

One lesson from this kernel: issue order matters more than you would expect.

- In our first version, each step began with the resident MMAs, then the streamed ones.
- `tcgen05.commit` waits for *all* MMAs that thread issued before it. So no ring slot could be freed, and no refill could be issued, until all 64 resident MMAs had finished. Meanwhile the copy engine sat idle.
- Consuming the stages that had already arrived first, and slipping 4 resident MMAs in after each refill, cut the step from 6.86 to 6.27 µs (−8.6%, same container).

Two more things we measured about B200's tensor cores:

- **A shared-memory operand holds the tensor pipe for the whole read.** At M = 128, N = 8, one MMA with A in shared memory takes 39 cycles per 4 KB of A. With A in TMEM it takes about 8 cycles. While the slow one runs, nothing else can use the tensor pipe, including `mma.sync` from other warps.
- **`tcgen05.cp` (shared memory → TMEM) is no way around this.** It runs at 64 cycles per 4 KB and holds the pipe too. TMEM pays off only for data that is written once and reused, like our resident table. It does not help for data that is used once, like the streamed part.

### End to end on B200

We have not run an end-to-end A/B on B200 yet; the kernels are not wired into SGLang there. What we have measured is the baseline's share. With Qwen3-4B and the γ = 16 drafter at bs = 1, the default walk is about 6.4% of a B200 round, against 11% on H100. That is because the faster L2 helps the walk more than it helps the target's verification. So expect a smaller end-to-end gain on B200 than on H100.

## Beyond DSpark: other serial heads

DSpark's Markov head is not the only serial head. Other drafters correct their block with a small recurrent cell that also ends in a vocabulary-wide matrix:

- **Domino:** a GRU cell that also reads the drafter's hidden states.
- **DeLS-Spec:** a GRU.
- **DSpark's own gated and RNN heads.**

They all have the same shape: a tiny state, and a big matrix that every step reads. So we turned the walk into a template, `walk_kernel<Cell>`, where each head supplies only its state update.

The question for each head is whether its recurrent matrix fits in the spare shared memory of every CTA:

- The gated head's 256 × 256 matrix does fit in int8, so every CTA computes the update redundantly and needs no extra exchange.
- The RNN head and the GRUs do not fit. Their state is spread across CTAs and gathered once or twice per step, at about 1.7 µs per gather.

H100, bs = 1, µs per round, against each head's own reference implementation. "Head only" means the round minus the lm_head GEMM, which both sides run identically:

| head | reference | γ = 7, ours / ref | γ = 15, ours / ref |
|---|---|---|---|
| DSpark gated (synthetic weights) | SGLang `GatedMarkovHead` | 62 / 410–412 (**6.6×**) | 97 / 874–877 (**9.0×**) |
| DSpark RNN (synthetic weights) | SGLang `RNNHead` | 69 / 480 (**7.0×**) | 113 / 1012–1013 (**8.9×**) |
| DeLS-Spec GRU (released weights) | the authors' CUDA-graph runner | 73–74 / 501–521 (**6.8–7.1×**) | 132–133 / 1066–1109 (**8.0–8.4×**) |
| Domino (released weights) | SGLang, exact full vocabulary | 68.5–68.8 / 487–488 (**7.1×**) | 126 / 1068–1072 (**8.5×**) |
| Domino (released weights) | SGLang default, 2,048-token candidate pool | 68.5–68.8 / 311–313 (**4.6×**) | 126 / 585–594 (**4.7×**) |
| DFlash2 (our 4B reproduction) | SGLang top-16 lattice walk | 39 / 78 (**2.0×**) | 64.5 / 77–84 (**1.2–1.3×**) |

- **DFlash2 is the exception.** It already restricts its serial part to a 16-token candidate lattice, which costs only ~0.6 µs per step. Our full-vocabulary walk costs ~3.2 µs per step, so it only wins on the fixed per-round cost, and the two cross near γ ≈ 21.
- **No public checkpoint uses DSpark's gated or RNN heads,** so those two rows use synthetic weights and measure time only.

**Domino end to end.** We hooked the kernel into SGLang's Domino path at bs = 1 (Qwen3-4B or Qwen3-8B with the released `Huang2020` Domino b16 drafters, temperature 0, concurrency 1, the same 605 prompts):

| target | vs SGLang exact (pool 0) | vs SGLang default (pool 2,048) |
|---|---|---|
| Qwen3-4B | **+16.8%** per round (saves 0.96 ms) | **+8.3%** (saves 0.48 ms) |
| Qwen3-8B | **+12.0%** (saves 1.03 ms) | **+6.6%** (saves 0.56 ms) |

- **The saving per round equals the walk saving.** For example, at 4B against the exact path, the round saves 0.96 ms and the walk microbenchmark saves 0.945 ms.
- **Outputs are identical.** With deterministic inference, the text is 605/605 identical to SGLang's exact path, at both sizes.
- **Acceptance is tied:** within 0.06%.

## Limitations

- **The upstream PR covers only part of this.** It targets H100 (sm_90a), vanilla Markov heads (rank 256), and bs ≤ 64. Under TP > 1 it falls back to the default walk.
- **The rest is not released yet.** The B200 kernels and the serial-head template are not public. Their end-to-end numbers are limited to what is reported above.
- **The gain shrinks as the target grows.** The saving per round is fixed, so a longer round means a smaller share.

## Code and timeline

- **Repository:** [github.com/tie-pilot-qxw/dspark-markov-walk](https://github.com/tie-pilot-qxw/dspark-markov-walk), public since **October 4, 2026, 16:25 PDT**. It contains the H100 kernels, the SGLang reference hook, the bf16 tensor-parallel kernels (`experimental/bf16_tp/`), tests, and the end-to-end A/B harness.
- **Upstream PR:** [sgl-project/sglang#42855](https://github.com/sgl-project/sglang/pull/42855), "[Spec][DSpark] Opt-in fused int8 Markov walk kernels (Hopper)", opened **October 6, 2026, 19:14 PDT**. It is opt-in behind `SGLANG_DSPARK_OPT_INT8_MARKOV_WALK=1`.

## Acknowledgments

We thank:

- the DSpark authors, for the drafter design and the released checkpoints;
- the SGLang team;
- the authors of Domino, DeLS-Spec and DFlash2;
- the authors of the public drafters used here (`shreybirmiwal`, `stepnivlk`, `Huang2020`).

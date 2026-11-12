# Cantonese Whisper quantization comparison

## Matched local measurements (2026-11-12)

| Variant | Weight MB | Raw corpus CER | Normalized corpus CER | Median decode s | RTF | MLX peak MiB | OS peak footprint MiB | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4-bit g64 | 877.696 | 29.04% | 14.92% | 0.661 | 0.0712 | 1672.3 | 2046.3 | 1098.0 |
| 3-bit g64 | 685.896 | 40.82% | 24.50% | 0.701 | 0.0771 | 1492.4 | 1897.9 | 918.3 |
| 2-bit g64 | 494.096 | 100.00% | 100.00% | 2.396 | 0.2580 | 1306.5 | 1595.2 | 734.1 |
| 2-bit g32 | 589.996 | 225.88% | 222.27% | 2.417 | 0.2598 | 1394.5 | 1682.3 | 821.1 |

The table uses the first prompted run for every variant, not the fastest repeat. MB is decimal bytes / 1,000,000; MiB is bytes / 1,048,576. Weight sizes are actual file lengths. Runtime files (weights plus config) occupy 877,703,168, 690,868,224, and 503,898,112 allocated bytes on this SSD for 4-, 3-, and 2-bit g64 respectively; filesystem allocation is not portable. The 2-bit g32 diagnostic weighs 589,996,337 bytes and is also unusable.

The 23 completed private Engineer Mode takes total 214.133 seconds and contain 20 distinct reading prompts in one Cantonese speaker's voice, with Cantonese and English technical speech. References are the prompts stored before recording, not newly human-verified verbatim transcriptions. There are repeated takes of two prompts, known number-vs-spoken-word mismatches, and one historically truncated recording. They are retained for comparison with the existing 4-bit evaluation. This is a small personal benchmark, not a population estimate or Common Voice test result. Recordings, references, transcripts, personal dictionaries, session IDs and private source passages are not distributed.

Every model used the same WAV bytes (hash-checked), mono 16 kHz PCM scaled by 1/32768, language `yue`, task `transcribe`, temperature 0, `condition_on_previous_text=False`, `word_timestamps=False`, `fp16=True`, sample length 224 per decoding window, and an explicit 64 MiB MLX cache limit. Prompted arms shared the same fixed InFlow mixed-Cantonese/English prompt. No personal dictionary, dynamic prompt context, hotwords, script correction, prompt-echo removal, silence gate, or LLM rewriting was applied. The prompt is kept with the private local experiment; no transcript or private vocabulary is being published. These are raw ASR measurements, not end-to-end dictation latency. Default temperature fallback was disabled in all arms.

Raw CER is Levenshtein edits / reference characters, including punctuation, whitespace and case. Normalized CER applies NFKC, Unicode casefold, removes whitespace and Unicode punctuation, and uses macOS CoreFoundation `Traditional-Simplified` on **both** reference and hypothesis, then totals edits / total reference characters (898 normalized reference characters; 1078 raw characters). It does not rewrite Cantonese into Mandarin, collapse number words into digits, forgive homophones, or repair names. CER can exceed 100% due to insertions. The 2-bit g32 score above 200% is intentional, not a percentage bug.

For continuity, the nonstandard per-take symmetric clean score (edits / max reference/hypothesis length, averaged) is 0.13242 for prompted 4-bit, 0.21384 for 3-bit, 1.00000 for 2-bit g64, and 0.96606 for 2-bit g32. This companion score is not conventional corpus CER. The historical 4-bit card's 0.132 value remains a legacy score, not a claim that the corrected CER is 13.2%. Historical term/marker substring scores are not reused; capitalization, script normalization, duplicate terms and substring matches made those scores unreliable.

## Prompt sensitivity and repeats

With no prompt, normalized CER was **16.37% (4-bit)**, **18.26% (3-bit)**, and **100% (2-bit g64)**. Thus 3-bit was much closer to 4-bit without the static prompt. A paired bootstrap over 20 prompt clusters (5000 resamples; seed 20261112) estimates the 3-bit excess at +9.58 percentage points with the prompt (exploratory 95% interval +3.87 to +17.27), and +1.89 points without it (−0.24 to +4.08). The latter does **not** establish equivalence or a statistically clear degradation. No prompt was tuned after seeing results.

Prompted 3-bit had fewer normalized edits on 1 take, more on 15, and tied on 7. Both prompted 4-bit and prompted 3-bit produced byte-identical raw transcripts on a second fresh-process run. Their median decode times were 0.661/0.668 seconds and 0.701/0.705 seconds respectively. OS peak footprint varied from 1961.5–2046.3 MiB for 4-bit and 1846.9–1897.9 MiB for 3-bit. The table reports first runs consistently. The 2-bit no-prompt arm confirms the collapse is not dependent on the static prompt.

## Memory and timing method

Apple M2 Max, 32 GiB unified memory; macOS 27.2.1, Python 3.13.7, MLX 0.32.2, mlx-whisper 0.4.3, NumPy 2.5.2. One model per fresh process, sequential runs, synchronized MLX work, one 1-second silent warm-up before measured samples. No source production preferences, worker code or installed model were modified. No GPU benchmark or converter ran concurrently; this was an ordinary desktop session, not a thermally controlled laboratory or an 8 GB Mac test.

MLX peak is `mx.get_peak_memory()` since reset before load, including loading, warm-up and decodes. It is an allocator statistic, **not RSS**. Process peak RSS and OS peak physical footprint come separately from macOS `/usr/bin/time -l`; current RSS/footprint samples use `proc_pid_rusage`. Sampled values are not labeled peaks. These overlapping memory measures must not be added together. The OS footprint numbers show real memory savings, but not a whole-machine RAM guarantee.

| Variant | Model load s | Silence warm-up s | Python package imports s |
|---|---:|---:|---:|
| 4-bit | 0.341 | 2.834 | 1.138 |
| 3-bit | 0.084 | 2.900 | 1.187 |
| 2-bit | 0.105 | 2.726 | 0.924 |

These load times use the existing filesystem cache; they are not cold-boot measurements. RTF is summed decode seconds / summed audio seconds (lower is faster), excludes loading and warm-up, and is ASR-only. Peak memory includes loading and warm-up. Severe 2-bit output failure also took longer; fewer bits did not mean faster useful transcription.

## Separate daily-recording diagnostics

Twelve pre-existing real-session recordings total 605.107 seconds: four read-aloud full turns, four noisy recordings, and four historical quiet comparators. They were selected from existing sets before seeing candidate outcomes. The read-aloud references cover only passages inside full turns with unscripted pre/postamble; the other eight have no verified human transcript. **No CER is reported for these twelve.** Disagreement between models is not ground-truth accuracy.

| Variant | Takes | Decode seconds | RTF | MLX peak MiB | OS peak footprint MiB | Empty after normalization |
|---|---:|---:|---:|---:|---:|---:|
| 4-bit | 12 | 34.200 | 0.0565 | 1727.4 | 1987.0 | 0 |
| 3-bit | 12 | 38.195 | 0.0631 | 1547.4 | 1810.3 | 1 |
| 2-bit g64 | 12 | 77.297 | 0.1277 | 1361.5 | 1628.5 | 12 |

The 3-bit/4-bit mean normalized transcript disagreement was 21.86% across these twelve recordings. This can identify material differences for later human listening; it cannot identify which model is right. Segment-level token-count flags are incomplete indicators: zero flags does not prove no decoding-window cap or truncation. All raw outputs and segment metadata are retained locally.

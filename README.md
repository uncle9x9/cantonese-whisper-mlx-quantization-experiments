# Cantonese Whisper MLX quantization

**4-bit is recommended for the tested Cantonese workflow.** On 23 private Cantonese and Cantonese–English reading takes, 4-bit produced the lowest normalized character error rate (CER) and the fastest median decoding. The 3-bit model saved memory and came close to 4-bit accuracy without the fixed prompt. The 2-bit model failed on these recordings in both prompt conditions. These results do not establish which variant is best for every speaker or language.

| Hugging Face model | Weights | Normalized CER, prompted | Normalized CER, unprompted | Median decode, prompted |
|---|---:|---:|---:|---:|
| [4-bit, recommended](https://huggingface.co/WinsonLi/whisper-large-v3-cantonese-mlx-4bit) | 877.7 MB | 14.92% | 16.37% | 0.661 s |
| [3-bit, experimental](https://huggingface.co/WinsonLi/whisper-large-v3-cantonese-mlx-3bit) | 685.9 MB | 24.50% | 18.26% | 0.701 s |
| [2-bit, research artifact](https://huggingface.co/WinsonLi/whisper-large-v3-cantonese-mlx-2bit) | 494.1 MB | 100.00% | 100.00% | 2.396 s |

These are matched local measurements from one speaker on an Apple M2 Max with 32 GiB of unified memory. [BENCHMARK.md](BENCHMARK.md) covers evaluation methods, memory, repeat runs, and a separate unscored daily-recording set. [Aggregate results](aggregate-results.json) and [disk measurements](disk-measurements.json) are machine-readable.

## Reproduce the conversion

Use an Apple Silicon Mac. The recorded setup was Python 3.13.7, MLX 0.32.2, and mlx-whisper 0.4.3. Install the [tested direct package versions](requirements-tested.txt) and download the pinned [Cantonese source checkpoint](https://huggingface.co/khleeloo/whisper-large-v3-cantonese/tree/f48a890f78c7b6acf723f25d8c81e232ac7469ca). Allow disk and memory for the full-precision source and conversion.

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-tested.txt

hf download khleeloo/whisper-large-v3-cantonese \
  --revision f48a890f78c7b6acf723f25d8c81e232ac7469ca \
  --include 'config.json' 'model*.safetensors*' \
  --local-dir source-model

for bits in 2 3 4; do
  python convert.py --torch-name-or-path source-model \
    --mlx-path "models/whisper-cantonese-${bits}bit" --dtype float16 \
    --quantize --q-bits "$bits" --q-group-size 64
done
```

The converter refuses to overwrite an existing output directory. Source revisions and input and output checksums are recorded for the [2-bit](provenance-2bit.json) and [3-bit](provenance-3bit.json) artifacts. The reproduced 4-bit weights matched the baseline weights byte for byte.

## Evaluate your own recordings

The original audio, reference text, and exact fixed prompt are private. The published CER cannot be independently reproduced from this repository alone. The harness can run the same procedure on your own recordings and references.

Copy [the example manifest](manifest.example.json) to a local path and replace the placeholders. Use mono 16 kHz, signed 16-bit PCM WAV files. Paths can be absolute or relative to the manifest. Give each sample a unique ID and a group and reference-kind label. Supply a human transcript or the pre-existing text read aloud for CER, and use a null or absent reference for audio without verified text. An optional `audio_sha256` field locks the exact recording bytes.

```sh
mkdir -p private outputs
cp manifest.example.json private/manifest.json
# Edit private/manifest.json and place your WAV files at the referenced paths.

for bits in 2 3 4; do
  /usr/bin/time -l python benchmark.py \
    --model "models/whisper-cantonese-${bits}bit" \
    --manifest private/manifest.json \
    --mode unprompted --output "outputs/${bits}bit-unprompted.jsonl" \
    2>"outputs/${bits}bit-unprompted.time.txt"
done
```

Each command starts a fresh process, uses the same decoding settings, and writes per-sample JSONL and a summary beside it. Run the commands sequentially. The direct decoder leaves dictionary replacement, hotwords, script repair, prompt-echo removal, and other postprocessing off. Script conversion is used only in normalized scoring. To test your own fixed prompt, use `--mode prompted --prompt-file private/prompt.txt` with the same UTF-8 file for each model and fresh output names. Keep manifests, recordings, prompts, and outputs in locations you control. Per-sample JSONL contains decoded text.

```sh
python -B -m unittest -v test_benchmark.py
```

## Licenses

The experiment scripts are [MIT licensed](LICENSE). The Apple-derived converter retains its [upstream MIT notice](LICENSE-convert.py-MIT). The derived Whisper weights are available separately on Hugging Face under the upstream [Apache 2.0 license](LICENSE-weights-Apache-2.0).

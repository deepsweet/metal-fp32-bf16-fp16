# Metal FP32 Vs BF16 Vs FP16 benchmark

It all started when I noticed a Reddit claim that using FP16 instead of BF16 for LLM on M1/M2 Apple Silicon leads to a prompt processing speed boost. I empirically [verified](https://github.com/jundot/omlx/issues/604) that with oMLX benchmarks, under the assumption that M1/M2 simply do not natively support BF16 at the hardware level.

There was also a [pull request](https://github.com/huggingface/transformers/pull/40458) to the Transformers framework stating:

>On Apple M1 and M2 bfloat16 is emulated in software (by Apple's Metal framework) using float32 for hardware instead, meaning there is no performance benefit, so on a M1 or M2 you should rather use float16 or float32.

So, it's a software emulation?

To confuse things even more, M2 Max positively claims to support the corresponding ARM instruction:

```sh
sysctl -n hw.optional.arm.FEAT_BF16

1
```

M1 Pro returns `0`. But that's about CPU, not Metal on GPU.

Yet I still couldn't find hard evidence.

Hence, the scalar arithmetic throughput benchmark:

## Results

### Apple M1 Pro

| Type | min (ms) | avg (ms) | sd (ms) |  GFLOPS | GiB/s |
|------|---------:|---------:|--------:|--------:|------:|
| FP32 | 1234.395 | 1241.841 |  13.577 |  885.39 |  1.21 |
| BF16 | 1235.884 | 1243.260 |  13.586 |  884.38 |  0.60 |
| FP16 |  741.595 |  749.489 |  11.583 | 1467.02 |  1.00 |

### Apple M2 Max

| Type | min (ms) | avg (ms) | sd (ms) |  GFLOPS | GiB/s |
|------|---------:|---------:|--------:|--------:|------:|
| FP32 |  534.474 |  534.860 |   0.825 | 2055.70 |  2.80 |
| BF16 |  534.893 |  534.922 |   0.045 | 2055.46 |  1.40 |
| FP16 |  323.488 |  323.501 |   0.013 | 3398.79 |  2.32 |

Here BF16 has exactly the same execution time and GFLOPS as FP32, but half the throughput because it uses 2‑byte elements instead of 4‑byte.

Conclusion: BF16 is clearly software‑emulated via FP32 on M1 and M2 chips, FP16 offers a noticeable advantage.

### Apple M3 Pro

| Type | min (ms) | avg (ms) | sd (ms) |  GFLOPS | GiB/s |
|------|---------:|---------:|--------:|--------:|------:|
| FP32 |  586.288 |  591.953 |  10.406 | 1857.43 |  2.53 |
| BF16 |  644.993 |  652.101 |  13.944 | 1686.11 |  1.15 |
| FP16 |  626.088 |  637.447 |   9.340 | 1724.87 |  1.18 |

Here BF16 performs similarly to FP16, within measurement noise.

Conclusion: BF16 is natively hardware-accelerated on M3 and newer chips, FP16 offers no measurable advantage.

## Usage

```sh
xcrun swiftc benchmark.swift -O -o benchmark

./benchmark
```

If something goes wrong:

```sh
# if you have full Xcode app installed
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer

# if you have only "Command Line Tools for Xcode" installed
sudo xcode-select --switch /Library/Developer/CommandLineTools

# if you don't have either install the tools
xcode-select --install
```

## See also

- https://huggingface.co/collections/deepsweet/qwen36-35b-a3b
- https://huggingface.co/collections/deepsweet/qwen36-27b

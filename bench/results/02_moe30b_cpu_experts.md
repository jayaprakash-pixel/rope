| model                          |       size |     params | backend    | ngl |  n_cpu_moe |  fa | rpk |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------: | --: | --: | --------------: | -------------------: |
| qwen3moe 30B.A3B Q4_K - Medium |  17.28 GiB |    30.53 B | CUDA       |  99 |         48 |   1 |   0 |           pp512 |         14.55 ± 0.19 |
| qwen3moe 30B.A3B Q4_K - Medium |  17.28 GiB |    30.53 B | CUDA       |  99 |         48 |   1 |   0 |            tg64 |          2.41 ± 0.23 |
| qwen3moe 30B.A3B Q4_K - Medium |  17.28 GiB |    30.53 B | CUDA       |  99 |         48 |   1 |   1 |           pp512 |         14.56 ± 0.01 |
| qwen3moe 30B.A3B Q4_K - Medium |  17.28 GiB |    30.53 B | CUDA       |  99 |         48 |   1 |   1 |            tg64 |          2.36 ± 0.12 |

build: e0949d09b (12317)

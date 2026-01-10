
<br>

master<br>
[![Hydrography: Inference ↠ Long Short-Term Memory Models [repatterning/arc-rnn-lstm]](https://github.com/repatterning/arc-rnn-lstm-inference/actions/workflows/main.yml/badge.svg?branch=master)](https://github.com/repatterning/arc-rnn-lstm-inference/actions/workflows/main.yml)

develop<br>
[![Hydrography: Inference ↠ Long Short-Term Memory Models [repatterning/arc-rnn-lstm]](https://github.com/repatterning/arc-rnn-lstm-inference/actions/workflows/main.yml/badge.svg?branch=develop)](https://github.com/repatterning/arc-rnn-lstm-inference/actions/workflows/main.yml)


<br>
<br>

**Inference**

Via an applicable infrastructure set-up, e.g., via an Amazon Web Services EC2 (Elastic Compute Cloud) machine with <abbr title="Compute Unified Device Architecture">CUDA</abbr>[^1] graphics processing units:

```shell
docker pull ghcr.io/repatterning/arc-rnn-lstm-inference:master

docker run --rm --gpus all --shm-size=15gb -e AWS_DEFAULT_REGION={region.code} \
  NVIDIA_DRIVER_CAPABILITIES=all ghcr.io/repatterning/arc-rnn-lstm-inference:master \
    src/main.py --codes '...,...' --request ... && sudo shutdown
```

wherein

* --request: $\in {0, 1, 2, 3}$ &Rarr; $0$ inspection, $1$ latest models live, $2$ on-demand inference service, $3$ warning period inference
* If --request $\in {1, 2}$ $\rightarrow$
  > --codes: A comma-separated list of gauge-station-time-series identification codes

**and** (a) relevant authentication & authorisation settings must be in-place, (b) storage areas must be set-up & accessible.

<br>
<br>

<br>
<br>

<br>
<br>

<br>
<br>

[^1]: <a href="https://spawn-queue.acm.org/doi/10.1145/1365490.1365500" target="_blank">Compute Unified Device Architecture</a>

<br>
<br>

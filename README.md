
<br>

**Inference**

Within an applicable infrastructure set-up:

```shell
docker pull ghcr.io/repatterning/arc-rnn-lstm-inference:master

docker run --rm --gpus all --shm-size=15gb -e AWS_DEFAULT_REGION={region.code} \
  NVIDIA_DRIVER_CAPABILITIES=all ghcr.io/repatterning/arc-rnn-lstm-inference:master \
    src/main.py --codes '...,...' --request ... && sudo shutdown
```

wherein

* --codes: A comma-separated list of gauge-station-time-series identification codes
* --request: $\in {0, 1, 2, 3}$ &Rarr; $0$ inspection, $1$ latest models live, $2$ on-demand inference service, $3$ warning period inference

<br>
<br>

<br>
<br>

<br>
<br>

<br>
<br>

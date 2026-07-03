# 🍌 Cold Chain Brain

A sensor-fused fruit spoilage monitor running on an **Arduino UNO Q**. Four physical sensors feed three independent signals — one physics-based, two machine-learned — combined by a majority vote into a single FRESH/ROTTEN verdict, shown on a live web dashboard.

## Hardware

| Sensor | Measures | Pin |
|---|---|---|
| DHT22 | Temperature + humidity | Digital D2 |
| MQ-3 | Alcohol/ethanol vapor | Analog A0 |
| MQ-135 | Air quality (ammonia/VOCs) | Analog A1 |

MQ-3/MQ-135 need ~20s+ after power-on for their heater elements to stabilize before readings are meaningful.

## Architecture

```
sketch.ino (STM32, Zephyr RTOS)
  → Bridge.notify("record_reading", temp, humidity, alcohol, air_quality)
  → msgpack-rpc-router (Unix socket)
  → main.py (Linux side, Docker container)
      ├─ Q10/Arrhenius kinetics estimator      (physics, hand-set constants)
      ├─ Environmental ML classifier            (logistic regression, trained offline)
      ├─ Gas sensor ML classifier                (logistic regression, trained offline)
      └─ Majority vote + persistence debounce → final verdict
  → WebUI brick → REST API (/api/status, /api/history, /api/models, /api/fruit)
  → assets/index.html (live dashboard)
```

## The three signals

1. **Q10 / Arrhenius kinetics** — standard food-science spoilage kinetics. Not machine-learned: `rate_multiplier = q10 ^ ((temp - ideal_temp) / 10) * (1 + humidity_sensitivity * |humidity - ideal_humidity|)`, integrated over real elapsed time into a "life consumed" fraction per fruit.

2. **Environmental ML classifier** — logistic regression trained on [A Multi-Parameter Dataset for Machine Learning Based Fruit Spoilage Prediction in an IoT-Enabled Cold Storage System](https://data.mendeley.com/datasets/czz68d9fwj/1) (Mendeley Data, CC BY 4.0, 10,995 rows). Trained on only `Temp + Humidity + Fruit` — the subset our hardware can actually supply. **88% test accuracy.**

3. **Gas sensor ML classifier** — logistic regression trained on the [Food Freshness Electronic Nose Dataset](https://doi.org/10.5281/zenodo.17285312) (Zenodo, CC BY 4.0, 595,293 rows of raw MQ2-MQ135 readings across 9 fruits over a 5-day window). Trained on `MQ-3 + MQ-135 + Fruit`. **91% test accuracy.** *Caveat: the dataset's own MQ-135 sensor read up to ~2975, ours maxes at 1023 (different circuit/ADC scaling) — treat this signal as directional, not precisely calibrated, until validated against your own hardware.*

Both ML models were trained offline (scikit-learn, in a throwaway Docker container — never on the device itself) and deployed as plain Python arithmetic (`standardize → one-hot → dot product → sigmoid`) with **no ML library running on the device.** See `training/` for the scripts, raw datasets, and exported weights.

## Fusion: majority vote + persistence

Each signal casts a bad/fresh vote (a model abstains if it has no data for the currently selected fruit). More than half of the *available* votes must say "bad" for the instantaneous verdict to flip — with all 3 available, that's a 2-of-3 majority. This replaced an earlier naive OR-of-3, which let a single noisy sensor (e.g. MQ-3/MQ-135 warm-up drift) flip the whole system. The reported verdict then only updates once the raw vote holds steady for 5 consecutive readings (~10s), filtering out single-sample spikes.

## Fruits supported

| Fruit | Kinetics | Env ML | Gas ML |
|---|---|---|---|
| Banana | ✓ | ✓ | ✓ |
| Tomato | ✓ | ✓ | ✓ |
| Strawberry | ✓ | — | ✓ |
| Orange | ✓ | ✓ | — |

Switch fruit live via the dropdown on the dashboard — it resets the kinetics estimator, vote debouncer, and reading history.

## Dashboard

Live at `http://<device-ip>:7000`. Shows: current sensor readings with trend sparklines, a click-through analysis-pipeline flowchart (click any box for its implementation detail), per-model cards with live P(bad)/accuracy/dataset info, and the raw-vs-debounced verdict state.

## Running it

```
arduino-app-cli app start /path/to/cold-chain-brain     # first run
arduino-app-cli app restart /path/to/cold-chain-brain   # after code changes
arduino-app-cli app logs /path/to/cold-chain-brain --follow
```

## Project structure

```
sketch/         Arduino sketch (Zephyr/STM32 side)
python/         main.py - fusion logic, REST API
assets/         index.html - the dashboard
training/       training scripts, raw datasets, exported model weights
```

## Known caveats

- Gas sensor model's "Bad" label is a modeling assumption (day 4-5 of a 5-day window in the source dataset), not an independently verified ground truth.
- Gas thresholds/scale are not calibrated to this specific hardware's exact sensor units.
- This is a hackathon prototype, not a validated food-safety instrument.

import math
import time
from collections import deque

from arduino.app_bricks.web_ui import WebUI
from arduino.app_utils import App, Bridge, Logger

logger = Logger("Cold Chain Brain")

# Storage reference data per fruit, from postharvest handling guidelines:
# ideal_temp_c / ideal_humidity_pct: recommended storage condition.
# reference_shelf_life_days: shelf life (days) when stored at the ideal condition.
# q10: spoilage-rate multiplier for every 10C above the ideal temperature (Q10 kinetics).
# humidity_sensitivity: extra spoilage-rate multiplier per percentage-point of humidity deviation.
FRUIT_PROFILES = {
    "banana": {
        "ideal_temp_c": 13.0,
        "ideal_humidity_pct": 90.0,
        "reference_shelf_life_days": 14.0,
        "q10": 2.0,
        "humidity_sensitivity": 0.02,
    },
    "strawberry": {
        "ideal_temp_c": 2.0,
        "ideal_humidity_pct": 90.0,
        "reference_shelf_life_days": 6.0,
        "q10": 2.5,
        "humidity_sensitivity": 0.03,
    },
    "tomato": {
        "ideal_temp_c": 12.5,
        "ideal_humidity_pct": 90.0,
        "reference_shelf_life_days": 12.0,
        "q10": 2.0,
        "humidity_sensitivity": 0.02,
    },
    "orange": {
        "ideal_temp_c": 5.0,
        "ideal_humidity_pct": 88.0,
        "reference_shelf_life_days": 42.0,
        "q10": 2.0,
        "humidity_sensitivity": 0.015,
    },
}

# The set of selectable fruits is exactly FRUIT_PROFILES.keys() - kinetics
# always needs a profile (it can't abstain like the ML models can), so a fruit
# only becomes selectable once it has one. Per-fruit ML coverage varies -
# check /api/models or MODEL_INFO for which of env/gas actually has data for
# the current fruit; a missing signal simply abstains from the vote.
FRUIT = "banana"

# Two independent logistic regressions - "late fusion": each trained on a
# different real dataset, covering a different subset of our sensors, combined
# at decision time rather than forced into one model. Both export the same
# shape (feature_order/scaler/fruit_categories/weights/bias) and are evaluated
# by the same generic `predict_probability()` below.

# Model 1 - environmental conditions (DHT22): "A Multi-Parameter Dataset for
# Machine Learning Based Fruit Spoilage Prediction in an IoT-Enabled Cold
# Storage System", Mendeley Data, doi:10.17632/czz68d9fwj.1, CC BY 4.0.
# 10,995 readings of Fruit/Temp/Humid/Light/CO2 labeled Good/Bad. Trained using
# only Temp+Humid+Fruit (the subset our DHT22 + fruit selector can actually
# supply - no light or calibrated CO2 sensor), holding out 20% for testing:
# 88% accuracy (vs. 91% if Light+CO2 were also available).
ENV_SPOILAGE_MODEL = {
    "feature_order": ["Temp", "Humid", "fruit_Banana", "fruit_Orange", "fruit_Pineapple", "fruit_Tomato"],
    "scaler_mean": {"Temp": 23.83219645293315, "Humid": 93.53240109140519},
    "scaler_scale": {"Temp": 1.2309633629311363, "Humid": 2.9881485948099313},
    "fruit_categories": ["Banana", "Orange", "Pineapple", "Tomato"],
    "weights": [1.9777287868882463, 7.173871042898569, -0.9772903227555151, 0.41534881078658087, -1.4294721289109165, -0.18758372291591005],
    "bias": -2.305582052353393,
}

# Model 2 - gas sensors (MQ-3/MQ-135): "Food Freshness Electronic Nose
# Dataset", Zenodo doi:10.5281/zenodo.17285312, CC BY 4.0. 595,293 raw MQ2-
# MQ135 readings across 9 fruits sampled once a day for 5 days. This dataset
# has no independent freshness label, so "Bad" was defined here as day 4-5 of
# the 5-day window vs. day 1-3 "Good" - a modeling assumption, not a verified
# ground truth. Trained using only MQ3+MQ135+Fruit (what our hardware has):
# 91% accuracy on held-out data.
# CALIBRATION CAVEAT: their MQ3 range (53-941) roughly matches our 0-1023 ADC,
# but their MQ135 range (434-2975) exceeds our ADC's max, meaning their
# circuit/sensor was scaled differently from ours. Treat this signal as
# directional, not precisely calibrated, until validated against your own
# known-fresh vs. known-rotten readings.
GAS_SPOILAGE_MODEL = {
    "feature_order": ["MQ3", "MQ135", "fruit_Banana", "fruit_Blueberry", "fruit_Grape", "fruit_Green Apple", "fruit_Kiwi", "fruit_Pear", "fruit_Red Apple", "fruit_Strawberry", "fruit_Tomato"],
    "scaler_mean": {"MQ3": 238.27856263937477, "MQ135": 965.384124191049},
    "scaler_scale": {"MQ3": 176.8505579219654, "MQ135": 489.12784793109194},
    "fruit_categories": ["Banana", "Blueberry", "Grape", "Green Apple", "Kiwi", "Pear", "Red Apple", "Strawberry", "Tomato"],
    "weights": [0.5311613981380398, 9.457005875530196, 2.982989283766363, 2.5994485874216733, -4.218682915828688, 6.2059211669662915, 3.4419056430291715, 4.685711510801695, 5.516929746731702, -17.78796913823683, -5.099825818826323],
    "bias": -1.6813630706150866,
}


# Provenance/accuracy/implementation info for the dashboard's "click for
# detail" panels - describes the pipeline, never used in the prediction math
# itself. get_model_info() (not a static dict) so "kinetics.profile" always
# reflects whichever fruit is currently selected.
def get_model_info() -> dict:
    return {
        "kinetics": {
            "label": "Q10 / Arrhenius Kinetics",
            "type": "Physics-based (hand-set constants)",
            "inputs": ["Temperature", "Humidity"],
            "profile": FRUIT_PROFILES.get(FRUIT),
            "explanation": (
                "Standard food-science spoilage kinetics, not machine-learned. "
                "rate_multiplier = q10 ^ ((temp - ideal_temp) / 10) * "
                "(1 + humidity_sensitivity * |humidity - ideal_humidity|). "
                "Each reading multiplies the fruit's base spoilage rate "
                "(1 / reference_shelf_life_days) by this factor and integrates "
                "it over real elapsed time into a running 'life_consumed' "
                "fraction; life_consumed >= 1.0 means 'rotten' by this signal."
            ),
        },
        "env": {
            "label": "Environmental ML Classifier",
            "type": "Logistic regression",
            "inputs": ["Temperature", "Humidity", "Fruit"],
            "accuracy": 0.88,
            "rows": 10995,
            "dataset": "Mendeley Data - Multi-Parameter Fruit Spoilage Dataset",
            "dataset_url": "https://data.mendeley.com/datasets/czz68d9fwj/1",
            "license": "CC BY 4.0",
            "supported_fruits": ENV_SPOILAGE_MODEL["fruit_categories"],
            "explanation": (
                "Trained offline (scikit-learn LogisticRegression, 80/20 "
                "stratified split) on real temp/humid/fruit -> Good/Bad "
                "labels. Deployed as plain arithmetic: standardize inputs, "
                "one-hot the fruit, dot product with the learned weights, "
                "add bias, sigmoid - no ML library runs on the device."
            ),
        },
        "gas": {
            "label": "Gas Sensor ML Classifier",
            "type": "Logistic regression",
            "inputs": ["MQ-3 (alcohol)", "MQ-135 (air quality)", "Fruit"],
            "accuracy": 0.91,
            "rows": 595293,
            "dataset": "Food Freshness Electronic Nose Dataset",
            "dataset_url": "https://doi.org/10.5281/zenodo.17285312",
            "license": "CC BY 4.0",
            "supported_fruits": GAS_SPOILAGE_MODEL["fruit_categories"],
            "caveat": "Label = day 4-5 of a 5-day window (our assumption, not a verified ground truth). Their MQ-135 scale (434-2975) exceeds our board's 0-1023 ADC range - treat as directional, not precisely calibrated.",
            "explanation": (
                "Same training/deployment approach as the environmental "
                "classifier, but fit on 595,293 raw MQ3/MQ135 readings "
                "instead - the dataset had no freshness label, so 'Bad' was "
                "defined here as day 4-5 of a 5-day monitoring window."
            ),
        },
        "fusion": {
            "label": "Majority Vote + Persistence",
            "type": "Rule-based combination of the 3 signals above",
            "persistence_readings": VERDICT_PERSISTENCE,
            "explanation": (
                f"Each of the 3 signals casts a bad/fresh vote (a model "
                f"abstains - counts as neither - if it has no data for the "
                f"current fruit). More than half of the available votes must "
                f"say 'bad' for the instantaneous ('raw') verdict to be "
                f"rotten - so with all 3 available, that's a 2-of-3 majority, "
                f"replacing an earlier 1-of-3 OR that let any single noisy "
                f"sensor (e.g. MQ-3/MQ-135 warm-up drift) flip the whole "
                f"system. The reported verdict then only updates once that "
                f"raw vote holds steady for {VERDICT_PERSISTENCE} consecutive "
                f"readings (~{VERDICT_PERSISTENCE * 2}s at the sketch's 2s "
                f"cadence), filtering out single-sample spikes."
            ),
        },
    }

HISTORY_MAXLEN = 60  # ~2 minutes of readings at one every 2s
history = deque(maxlen=HISTORY_MAXLEN)

# How many consecutive readings a raw verdict must repeat before it's reported.
# At ~1 reading/2s from the sketch, 5 means ~10s of sustained agreement -
# filters out single-reading sensor noise (e.g. MQ-3/MQ-135 warm-up drift)
# from flipping the reported status instantly.
VERDICT_PERSISTENCE = 5


def predict_probability(model: dict, fruit: str, **numeric_features: float) -> float | None:
    """Evaluates a trained logistic regression (see ENV_/GAS_SPOILAGE_MODEL) as a
    plain dot product + sigmoid. Returns None if `fruit` isn't one it was trained on."""
    capitalized_fruit = fruit.title()  # "green apple" -> "Green Apple"; also fine for single-word fruits
    if capitalized_fruit not in model["fruit_categories"]:
        return None

    scaled = {name: (value - model["scaler_mean"][name]) / model["scaler_scale"][name] for name, value in numeric_features.items()}
    features = [scaled.get(name, 1.0 if name == f"fruit_{capitalized_fruit}" else 0.0) for name in model["feature_order"]]
    z = model["bias"] + sum(w * f for w, f in zip(model["weights"], features))
    return 1.0 / (1.0 + math.exp(-z))


class ShelfLifeEstimator:
    """Tracks cumulative spoilage using Q10 temperature kinetics plus a humidity penalty."""

    def __init__(self, profile: dict):
        self.profile = profile
        self.life_consumed = 0.0  # 0 = fresh, >=1 = rotten
        self._last_update = None
        self._last_rate_multiplier = 1.0

    def _rate_multiplier(self, temperature_c: float, humidity_pct: float) -> float:
        profile = self.profile
        temp_multiplier = profile["q10"] ** ((temperature_c - profile["ideal_temp_c"]) / 10.0)
        humidity_multiplier = 1.0 + profile["humidity_sensitivity"] * abs(humidity_pct - profile["ideal_humidity_pct"])
        return temp_multiplier * humidity_multiplier

    def update(self, temperature_c: float, humidity_pct: float):
        now = time.monotonic()
        if self._last_update is not None:
            elapsed_days = (now - self._last_update) / 86400.0
            base_rate_per_day = 1.0 / self.profile["reference_shelf_life_days"]
            self.life_consumed += base_rate_per_day * self._last_rate_multiplier * elapsed_days
        self._last_rate_multiplier = self._rate_multiplier(temperature_c, humidity_pct)
        self._last_update = now

    @property
    def is_rotten(self) -> bool:
        return self.life_consumed >= 1.0

    @property
    def remaining_days(self) -> float:
        base_rate_per_day = 1.0 / self.profile["reference_shelf_life_days"]
        current_rate = base_rate_per_day * self._last_rate_multiplier
        if current_rate <= 0:
            return float("inf")
        return max(0.0, 1.0 - self.life_consumed) / current_rate


class VerdictDebouncer:
    """Requires a raw boolean verdict to repeat for `persistence` consecutive
    updates before it's reflected in `.current` - smooths out single-reading
    spikes so one noisy sample can't instantly flip the reported status."""

    def __init__(self, persistence: int):
        self.persistence = persistence
        self.current = False
        self._streak_value = None
        self._streak_len = 0

    def update(self, raw_value: bool) -> bool:
        if raw_value == self._streak_value:
            self._streak_len += 1
        else:
            self._streak_value = raw_value
            self._streak_len = 1
        if self._streak_len >= self.persistence:
            self.current = raw_value
        return self.current


def compute_votes(temperature: float, humidity: float, alcohol_level: int, air_quality_level: int) -> dict:
    """Evaluates all three signals independently. Each is either True/False or
    None if that model has no data for the current FRUIT (abstains)."""
    env_bad_probability = predict_probability(ENV_SPOILAGE_MODEL, FRUIT, Temp=temperature, Humid=humidity)
    gas_bad_probability = predict_probability(GAS_SPOILAGE_MODEL, FRUIT, MQ3=alcohol_level, MQ135=air_quality_level)
    return {
        "kinetics_bad": estimator.is_rotten,
        "env_bad_probability": env_bad_probability,
        "env_bad": None if env_bad_probability is None else env_bad_probability >= 0.5,
        "gas_bad_probability": gas_bad_probability,
        "gas_bad": None if gas_bad_probability is None else gas_bad_probability >= 0.5,
    }


def majority_vote(votes: dict) -> bool:
    """More than half of the *available* votes (models that abstain due to an
    unsupported fruit don't count either way) must say "bad". With all 3
    available this is a 2-of-3 majority - a single outlier signal (e.g. gas
    sensor warm-up drift) can no longer flip the verdict alone, unlike the
    previous OR-of-3 logic."""
    ballots = [v for v in (votes["kinetics_bad"], votes["env_bad"], votes["gas_bad"]) if v is not None]
    if not ballots:
        return False
    return sum(ballots) * 2 > len(ballots)


estimator = ShelfLifeEstimator(FRUIT_PROFILES[FRUIT])
verdict_debouncer = VerdictDebouncer(VERDICT_PERSISTENCE)
latest_reading = None
latest_gas_reading = None


def record_reading(celsius: float, humidity: float, alcohol_level: int, air_quality_level: int):
    """Called by the sketch (via Bridge.notify) every time a new sensor reading is available."""
    global latest_reading, latest_gas_reading
    latest_reading = (celsius, humidity)
    latest_gas_reading = (alcohol_level, air_quality_level)
    estimator.update(celsius, humidity)
    # Advance the debounce state machine exactly once per physical reading -
    # get_status() must not do this itself, since it may be called more or
    # less often than readings actually arrive (e.g. dashboard polling).
    raw_rotten = majority_vote(compute_votes(celsius, humidity, alcohol_level, air_quality_level))
    verdict_debouncer.update(raw_rotten)
    snapshot = get_status()
    snapshot["t"] = time.time()
    history.append(snapshot)


Bridge.provide("record_reading", record_reading)

ui = WebUI()


def get_status():
    if latest_reading is None:
        return {"ready": False, "fruit": FRUIT}

    temperature, humidity = latest_reading
    alcohol_level, air_quality_level = latest_gas_reading
    votes = compute_votes(temperature, humidity, alcohol_level, air_quality_level)
    raw_rotten = majority_vote(votes)

    return {
        "ready": True,
        "fruit": FRUIT,
        "temperature": round(temperature, 1),
        "humidity": round(humidity, 1),
        "alcohol_level": alcohol_level,
        "air_quality_level": air_quality_level,
        "kinetics_life_consumed_pct": round(min(estimator.life_consumed, 1.0) * 100, 1),
        "kinetics_predicts_bad": votes["kinetics_bad"],
        "env_bad_probability": round(votes["env_bad_probability"], 3) if votes["env_bad_probability"] is not None else None,
        "env_predicts_bad": bool(votes["env_bad"]),
        "gas_bad_probability": round(votes["gas_bad_probability"], 3) if votes["gas_bad_probability"] is not None else None,
        "gas_predicts_bad": bool(votes["gas_bad"]),
        # raw_rotten: instantaneous 2-of-3 majority vote, no smoothing.
        # rotten: debounced - only flips after `raw_rotten` holds steady for
        # VERDICT_PERSISTENCE consecutive readings. This is what's shown as
        # the headline verdict.
        "raw_rotten": raw_rotten,
        "rotten": verdict_debouncer.current,
        "remaining_days": round(estimator.remaining_days, 1),
    }


def get_fruit_options():
    return {"current": FRUIT, "available": sorted(FRUIT_PROFILES.keys())}


def set_fruit(fruit: str):
    """Switches the active fruit. Resets the kinetics estimator, the vote
    debouncer, and the reading history, since all three are specific to
    whichever fruit was previously selected."""
    global FRUIT, estimator, verdict_debouncer
    fruit = fruit.lower()
    if fruit not in FRUIT_PROFILES:
        return {"error": f"Unknown fruit '{fruit}'. Choose one of {sorted(FRUIT_PROFILES.keys())}"}
    FRUIT = fruit
    estimator = ShelfLifeEstimator(FRUIT_PROFILES[FRUIT])
    verdict_debouncer = VerdictDebouncer(VERDICT_PERSISTENCE)
    history.clear()
    logger.info(f"Switched active fruit to '{FRUIT}'")
    return get_fruit_options()


ui.expose_api("GET", "/api/status", get_status)
ui.expose_api("GET", "/api/history", lambda: list(history))
ui.expose_api("GET", "/api/models", get_model_info)
ui.expose_api("GET", "/api/fruit", get_fruit_options)
ui.expose_api("POST", "/api/fruit", set_fruit)


def loop():
    time.sleep(5)
    status = get_status()
    if not status["ready"]:
        logger.info("Waiting for sensor data...")
        return

    state = "ROTTEN" if status["rotten"] else "fresh"
    raw_state = "rotten" if status["raw_rotten"] else "fresh"
    env_info = f"env_p(bad)={status['env_bad_probability']}" if status["env_bad_probability"] is not None else "env=n/a"
    gas_info = f"gas_p(bad)={status['gas_bad_probability']}" if status["gas_bad_probability"] is not None else "gas=n/a"
    logger.info(
        f"{FRUIT}: {status['temperature']}C, {status['humidity']}% RH, "
        f"alcohol={status['alcohol_level']}, air_quality={status['air_quality_level']}, {env_info}, {gas_info} "
        f"(raw vote: {raw_state}) -> {state}, estimated shelf life remaining: {status['remaining_days']} days"
    )


App.run(user_loop=loop)

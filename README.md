# cold-chain-brain

## Fruit Freshness Status

A static, no-backend dashboard that visualizes real-time shelf-life intelligence for produce, using a banana as the example fruit. Built as the front-end mockup for **Cold Chain Brain**, a sensor-driven post-harvest spoilage monitoring system (see `cold-chain-brain-build-guide.docx` for the full hardware/ML build plan).

![status](https://img.shields.io/badge/status-static%20mockup-blue) ![stack](https://img.shields.io/badge/stack-HTML%2FCSS%2FJS-informational)

## What this is

An enterprise-style monitoring dashboard, split into:

- **Produce Visual Inspection** — a procedurally generated SVG banana (curve + taper function, not a static drawing) that shows freshness state through color and speckling, plus a radial gauge for predicted days remaining, a confidence bar, and a raw → ripe → use soon → spoiled timeline.
- **Live Sensor Feed** — a data table of 8 mocked sensor channels (temperature, humidity, ethanol, VOC/air quality, hydrogen sulphide, CO₂, ripeness index, spoilage risk), each with a current reading, ideal range, mini trend chart, and status.
- **Prediction Rationale** — a plain-language explanation of the shelf-life estimate, generated from the same mock sensor values shown in the table.

All data is mocked in the browser and gently randomized on an interval to feel "live." There is no backend, no build step, and no external API calls beyond Google Fonts.

## Running locally

No dependencies or install step — it's a static site.

```bash
cd ccb
python3 -m http.server 8787
```

Then open **http://localhost:8787**.

(Any static file server works — `npx serve`, VS Code's Live Server, etc. — this just avoids opening `index.html` directly via `file://`, which some browsers restrict for module/font loading.)

## Project structure

```
ccb/
├── index.html   # page structure: masthead, hero panel, sensor table, rationale card
├── style.css    # design system: tokens, layout, table, gauge, panel chrome
├── script.js    # mock sensor data, procedural banana SVG, live-update simulation
└── cold-chain-brain-build-guide.docx   # hardware/ML build plan this UI is mocked for
```

## Customizing the mock scenario

All sensor state lives in the `SENSORS` array at the top of `script.js` — one object per row:

```js
{
  id: "temperature",
  label: "Temperature",
  unit: "°C",
  value: 27.8,          // current reading
  scaleMin: 5, scaleMax: 40,     // full gauge range
  idealMin: 12, idealMax: 18,    // "ideal" band drawn on the mini bar
  idealText: "12–18°C",
  status: "Warning",     // text shown in the status column
  tier: "warn",           // "optimal" | "warn" | "critical" — drives color
  jitter: 0.12,           // how much the value wanders per tick
}
```

Edit `value`, `status`, and `tier` directly — the mini bar, sparkline, and the "Prediction Rationale" card all read from this array, so they stay in sync automatically.

To change the headline freshness state (days left, gauge color, "Use Soon" tag, recommendation copy, timeline position), edit:
- `index.html` — the `.status-tag`, `.recommendation-text`, and `.timeline` markup
- `script.js` — the `renderGauge(2)` call in `init()` (first argument is days remaining)

To change how ripe/spotted the banana looks, tune the constants in `script.js`:
- `BANANA_CURVE` — the four control points defining the body's centerline
- `bodyHalfWidth()` — the taper profile (thin ends, fat middle)
- `sampleFreckleT()` / `buildFreckles()` — speckle density and placement bias toward the ends

Live-update cadence is controlled by the `setInterval(...)` calls at the bottom of `init()`.

## Wiring up real sensors

Everything in `script.js` under `simulateSensorTick()` is a placeholder. The build guide's architecture serves live readings from a local Flask dashboard (`dashboard.py`) on the device's Linux side. To connect real data, replace the mock interval with a subscription, e.g.:

```js
const ws = new WebSocket("ws://cold-chain-brain.local/sensors");
ws.onmessage = (evt) => {
  const reading = JSON.parse(evt.data);
  // update the matching SENSORS entry, then call updateSensorRow(sensor)
};
```

See the `REAL SENSOR HOOK` comment in `script.js` for the exact spot.

## Browser support notes

- Uses CSS `color-mix()` for status tinting — needs a recent browser (Safari 16.2+, Chrome 111+, Firefox 113+).
- Loads IBM Plex Mono + Inter from Google Fonts; falls back to system fonts if offline.

# Hero IntelliInspect

**AI-Powered Vehicle Damage Detection & Intelligent Assessment** — built with **Streamlit** + **Ultralytics YOLO**.

Upload a vehicle image to detect damages, estimate severity, and generate an instant (AI-generated) repair cost breakdown.

---

## Features

- **Damage detection (YOLO)**: Detects common vehicle damage classes.
- **Severity assessment**: Computes a severity score + Low/Medium/High level.
- **Cost estimation**: Estimates repair cost based on detected damage types.
- **Beautiful interactive report**: “Detected Damages”, severity analysis, and cost breakdown.
- **Confidence-aware insights**: Displays per-damage detection confidence.

---

## Supported Damage Classes

The model outputs (via YOLO) the following damage labels (as used by the app):

- `glass shatter`
- `lamp broken`
- `tire flat`
- `dent`
- `crack`
- `scratch`

---

## Disclaimer

This project provides **AI-generated** damage detection and **estimated** repair costs. Final repair quotes and insurance claim decisions require **physical inspection** and review under relevant policy terms.

---

## License

This project is licensed under the **MIT License**. See [LICENSE](./LICENSE).


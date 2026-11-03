# platelog CLI

A small Python demo that detects trucks in video footage and reads their license plates using YOLO and Fast-ALPR.

The pipeline tracks trucks across sampled video frames, collects license plate observations, and uses majority voting to determine the most likely plate. Each detected truck is saved as an event with its plate, confidence, timestamps, and best frame.

## Installation

Clone the repository:

```bash
git clone https://github.com/PockyStiks/platelog.git
cd platelog
```

### Option 1: Python virtual environment (venv)

Create and activate a virtual environment:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -e .
```

### Option 2: Conda

Create and activate a Conda environment:

```bash
conda create -n platelog python=3.13
conda activate platelog
```

Install the project:

```bash
pip install -e .
```

## Usage

Run the pipeline by providing a folder containing video clips:

```bash
python platelog.py input/
```

By default, results are saved in the `results/` directory.

You can specify a different output directory using `--out`:

```bash
python platelog.py input/ --out my_results/
```

For a list of available arguments:

```bash
python platelog.py --help
```

## Output

The output directory contains:

```text
results/
├── events.json
├── RESULTS.md
└── frames/
    ├── IMG_5177_truck1.jpg
    ├── IMG_5178_truck1.jpg
    └── ...
```

- **events.json** – Detected trucks and their license plate readings, timestamps, confidence scores, and best frame paths.
- **RESULTS.md** – Evaluation results comparing detected plates against the manually defined ground truth, including truck detection and plate recognition metrics.
- **frames/** – Best image for each detected truck. A truck image is saved when its plate is unreadable; otherwise, the best license plate crop is saved.

## Configuration

Detection and recognition settings, such as confidence thresholds, frame sampling rate, and minimum agreeing OCR reads, can be adjusted in the settings section at the beginning of `platelog.py`.

The ground truth used for evaluation is also defined in `platelog.py`.

## Tests

Tests are written using `pytest` and focus on the pipeline's decision-making logic, including plate recognition, conflicting OCR readings, truck tracking, and image selection.

Run all tests from the project root:

```bash
pytest -v
```

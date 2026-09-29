# Road Damage Detection and Severity Assessment

A lightweight computer vision system that looks at a photo of a road and:

1. **Detects** whether there is road damage
2. **Localizes** it with a bounding box
3. **Classifies** it as a pothole, longitudinal crack, transverse crack or alligator crack
4. **Estimates severity** as Low, Medium or High

It is built with TensorFlow and MobileNetV2 and comes with a Streamlit web app.

## How it works

```
Road photo
   │
   ▼
MobileNetV2 (transfer learning)  ──►  Damage type + confidence
   │
   ▼
Grad-CAM heatmap (where the model looked)
   │
   ├──►  Largest hot region  ──►  Bounding box
   └──►  Share of photo covered  ──►  Severity (Low / Medium / High)
```

**Transfer learning.** MobileNetV2 was already trained on over a million ImageNet photos, so it
knows edges, textures and shapes. We keep that knowledge and train only a small new layer for
our four classes (stage 1), then gently fine-tune the last 30 layers (stage 2).

**Grad-CAM.** After a prediction, Grad-CAM measures which parts of the image pushed the model
towards its answer. The strongest area becomes the bounding box.

**Severity.** Severity is estimated from how much of the photo the damaged area covers:

| Severity | Damaged area covers |
|----------|---------------------|
| Low      | less than 15% of the photo |
| Medium   | 15% to 35% |
| High     | more than 35% |

These thresholds are set in `config.py` and can be tuned.

## Dataset and augmentation

The dataset has 100 images per class. Each class is split 70 / 15 / 15 into train, validation
and test sets. Only the training images are augmented, with 4 extra copies each:

| Augmentation | Setting |
|--------------|---------|
| Tilt (perspective) | corners moved up to 12% inward |
| Rotation | −15° to +15° |
| Horizontal flip | 50% chance |
| Brightness | 50% to 120% of original |

Rotation is kept small on purpose: a 90° turn would make a longitudinal crack look like a
transverse one and teach the model the wrong label.

## Project structure

```
road-damage-detection/
├── app.py              Streamlit web app
├── analyzer.py         Prediction, Grad-CAM box and severity
├── config.py           All settings in one place
├── prepare_data.py     Splits and augments the dataset
├── train.py            Trains and evaluates the model
├── requirements.txt
├── .streamlit/config.toml   App colours
├── data/raw/           Your images go here (one folder per class)
├── models/             Trained model is saved here
└── results/            Training curves, confusion matrix, test report
```

## Setup (VS Code)

You need **Python 3.11 or 3.12** (TensorFlow may not support the newest Python release yet).

1. Open this folder in VS Code (**File → Open Folder**).
2. Open a terminal (**Terminal → New Terminal**) and create a virtual environment:

   ```bash
   python -m venv .venv
   ```

3. Activate it:

   - Windows: `.venv\Scripts\activate`
   - macOS / Linux: `source .venv/bin/activate`

   You should see `(.venv)` at the start of the terminal line. On Windows, if PowerShell
   blocks the script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and try again.

4. Install the libraries:

   ```bash
   pip install -r requirements.txt
   ```

5. Put your images into `data/raw/`, one folder per class:

   ```
   data/raw/
   ├── alligator_crack/
   ├── longitudinal_crack/
   ├── pothole/
   └── transverse_crack/
   ```

## Usage

```bash
python prepare_data.py      # split + augment
python train.py             # train + evaluate (see results/)
streamlit run app.py        # open the web app
```

On a normal laptop CPU, training usually takes around 20 to 60 minutes.

## Results

After training, add your numbers and charts here:

- Test accuracy: _fill in from the terminal output_
- `results/training_curves.png`
- `results/confusion_matrix.png`
- `results/test_report.txt`

## Limitations

- Bounding boxes come from Grad-CAM, not a trained object detector, so they are approximate.
- The app marks one defect per photo, even if a photo contains several.
- Severity is a rule based on area, not a learned prediction from labelled severity data.
- The model always picks one of the four classes, even for photos with no damage.

## Future work

- Label bounding boxes and severity (for example with makesense.ai or Roboflow) and train a
  MobileNetV2 model with separate outputs for class, box and severity.
- Add a "no damage" class.
- Train a full object detector (such as SSD-MobileNetV2) to find several defects per photo.

## Tech stack

Python · TensorFlow / Keras · MobileNetV2 · OpenCV · scikit-learn · Streamlit

# Adding the real YOLO model

This project currently uses a deterministic dummy model so the interface can be tested before the trained model is ready.

## Expected model type

The app expects an **Ultralytics YOLO classification model** that outputs probabilities over source-type classes, for example:

```text
tube_well_or_borehole
protected_dug_well
unprotected_dug_well
surface_water
rainwater_harvesting
...
```

The app then:

1. Sorts source-type probabilities from highest to lowest.
2. Shows the top 3 source-type predictions.
3. Takes the top source-type prediction only.
4. Maps that source type to `improved` or `non-improved` using `SOURCE_TYPE_TO_BINARY_CLASS` in `app/config.py`.

The binary prediction is **not** computed by summing all improved and non-improved probabilities.

## Expected model path

Place the trained model here:

```text
app/model/water_source_yolo.pt
```

Then edit:

```text
app/config.py
```

Change:

```python
MODEL_BACKEND = "dummy"
```

To:

```python
MODEL_BACKEND = "yolo"
MODEL_VERSION = "yolo-v1"
```

## Update the source-type mapping

In `app/config.py`, update:

```python
SOURCE_TYPE_TO_BINARY_CLASS = {...}
```

The keys must match the model's class names after normalization. The app normalizes class names by:

- lowercasing
- replacing spaces, hyphens, and slashes with underscores
- removing special characters

For example, these model labels all normalize similarly:

```text
Tube Well or Borehole -> tube_well_or_borehole
Tube-well/Borehole    -> tube_well_borehole
```

If a top source type is missing from the mapping, the app will show `Unknown improved / non-improved category`.

## Install YOLO dependencies

For local development:

```bash
pip install -r requirements-yolo.txt
```

Then run:

```bash
streamlit run app/app.py
```

## Packaging note

When the YOLO model is added, install the extra requirements during packaging. In the GitHub Actions workflow, uncomment the YOLO dependency installation step, or add:

```bash
pip install -r requirements-yolo.txt
```

The generated installer will be much larger because recent YOLO versions usually depend on PyTorch-related packages.

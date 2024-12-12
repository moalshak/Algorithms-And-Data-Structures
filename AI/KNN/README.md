
# How to run

```bash
virtualenv .venv
source .venv/bin/activate
pip install manim numpy
```

```bash
manim -pql -qh knn-fruit-classifier.py FruitClassification
mv media/videos/knn-fruit-classifier/1080p60/FruitClassification.mp4 .
```

The KNN visualization is based on the file `knn.py`.

```bash
manim -pql -qh knn-general.py KNNVisualization
mv media/videos/knn-general/1080p60/KNNVisualization.mp4 .
```
from typing import List, Dict, Any
import numpy as np
from collections import Counter
from pydantic import BaseModel, Field, field_validator, PrivateAttr
from helper import *

class KNN(BaseModel):
    k: int = Field(title="Number of neighbors", default=3)
    _parameters: dict = PrivateAttr(default_factory=dict)
    
    @field_validator("k")
    def k_greater_than_zero(cls, value):
        if value <= 0:
            raise ValueError("k must be greater than 0")

    def fit(self, observations:np.ndarray, ground_truth: np.ndarray) -> None:
        self._parameters = {"observations": observations, "ground_truth": ground_truth}
    
    def predict(self, observations: np.ndarray) -> np.ndarray:
        if self._parameters is None:
            raise ValueError("Model not fitted. Call 'fit' with appropriate arguments before using 'predict'.")
        
        predictions = [self._predict_single(x) for x in observations]
        return np.array(predictions)
    
    def _predict_single(self, observations: np.ndarray) -> Any:
        distances = np.linalg.norm(observations - self._parameters["observations"], axis=1)
        sorted_indices = np.argsort(distances)
        k_indices = sorted_indices[:self.k]
        k_nearest_labels = [self._parameters["ground_truth"][i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

class FruitClassifier(KNN):
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        return super().fit(observations, ground_truth)
    
    def ant():
        pass

# Generate some random fruits
fruits, labels, apples, oranges = generate_fruits()

# Create and train the model
fruit_classifier = FruitClassifier(k=3)
fruit_classifier.fit(fruits, labels)

new_fruit = gen_random_fruit()

prediction = fruit_classifier.predict(new_fruit)
print(f"The new fruit is predicted to be: {prediction[0]}")

plot_fruits(apples, oranges, new_fruit, prediction)
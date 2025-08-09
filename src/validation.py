# src/validation.py
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import numpy as np


class IrisFeatures(BaseModel):
    """Pydantic model for Iris dataset features"""

    sepal_length: float = Field(..., gt=0, le=10, description="Sepal length in cm")
    sepal_width: float = Field(..., gt=0, le=10, description="Sepal width in cm")
    petal_length: float = Field(..., gt=0, le=10, description="Petal length in cm")
    petal_width: float = Field(..., gt=0, le=10, description="Petal width in cm")

    @validator("*")
    def validate_numeric_range(cls, v, field):
        """Validate that all features are within reasonable ranges"""
        if not isinstance(v, (int, float)):
            raise ValueError(f"{field.name} must be numeric")
        if v < 0:
            raise ValueError(f"{field.name} must be positive")
        if v > 20:  # Reasonable upper bound
            raise ValueError(f"{field.name} seems unreasonably large")
        return float(v)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model prediction"""
        return np.array(
            [self.sepal_length, self.sepal_width, self.petal_length, self.petal_width]
        ).reshape(1, -1)


class HousingFeatures(BaseModel):
    """Pydantic model for California Housing features"""

    median_income: float = Field(..., gt=0, description="Median income")
    house_age: float = Field(..., ge=0, le=100, description="House age in years")
    avg_rooms: float = Field(..., gt=0, description="Average rooms per household")
    avg_bedrooms: float = Field(..., gt=0, description="Average bedrooms per household")
    population: float = Field(..., gt=0, description="Population")
    avg_occupancy: float = Field(..., gt=0, description="Average occupancy")
    latitude: float = Field(..., ge=32, le=42, description="Latitude")
    longitude: float = Field(..., ge=-125, le=-114, description="Longitude")

    @validator("avg_bedrooms", "avg_rooms")
    def validate_room_ratios(cls, v, field):
        """Validate room-related features"""
        if v > 20:  # Reasonable upper bound
            raise ValueError(f"{field.name} seems unreasonably large")
        return v

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model prediction"""
        return np.array(
            [
                self.median_income,
                self.house_age,
                self.avg_rooms,
                self.avg_bedrooms,
                self.population,
                self.avg_occupancy,
                self.latitude,
                self.longitude,
            ]
        ).reshape(1, -1)


class PredictionRequest(BaseModel):
    """Generic prediction request with validation"""

    features: IrisFeatures  # or HousingFeatures

    class Config:
        schema_extra = {
            "example": {
                "features": {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2,
                }
            }
        }

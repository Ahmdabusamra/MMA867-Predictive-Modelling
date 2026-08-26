#!/usr/bin/env python3
"""Train-fitted missing-data preprocessing for the Ames House Prices data.

Run from the repository root:
    python3 src/preprocess_missing_data.py

The script never modifies the source CSV files.  It writes processed copies to
``data/processed`` and verifies that all missing values have been resolved.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_TRAIN = ROOT / "train.csv"
INPUT_TEST = ROOT / "test.csv"
OUTPUT_DIR = ROOT / "data" / "processed"

ABSENCE_LABELS = {
    "Alley": "NoAlley",
    "FireplaceQu": "NoFireplace",
    "GarageType": "NoGarage",
    "GarageFinish": "NoGarage",
    "GarageQual": "NoGarage",
    "GarageCond": "NoGarage",
    "PoolQC": "NoPool",
    "Fence": "NoFence",
    "MiscFeature": "NoMiscFeature",
}
BASEMENT_CATEGORICAL = ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]
BASEMENT_NUMERIC = ["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath"]
GENERIC_CATEGORICAL = ["Electrical", "MSZoning", "Utilities", "Exterior1st", "Exterior2nd", "KitchenQual", "SaleType"]


class MissingDataPreprocessor:
    """Fits imputation statistics on training predictors only."""

    def fit(self, train: pd.DataFrame) -> "MissingDataPreprocessor":
        features = train.drop(columns=["SalePrice"], errors="ignore")
        self.numeric_medians_ = features.median(numeric_only=True).to_dict()
        self.neighborhood_medians_ = {
            column: features.groupby("Neighborhood")[column].median().to_dict()
            for column in ("LotFrontage", "MasVnrArea")
        }
        self.garage_type_medians_ = {
            column: features.groupby("GarageType")[column].median().to_dict()
            for column in ("GarageYrBlt", "GarageCars", "GarageArea")
        }
        return self

    @staticmethod
    def _fill_by_group(
        frame: pd.DataFrame, column: str, group: str, group_medians: dict, fallback: float
    ) -> None:
        missing = frame[column].isna()
        frame.loc[missing, column] = frame.loc[missing, group].map(group_medians)
        frame[column] = frame[column].fillna(fallback)

    @staticmethod
    def _no_basement(frame: pd.DataFrame) -> pd.Series:
        # A missing total paired with a missing quality is the one incomplete
        # no-basement record in the supplied test set; do not classify a
        # positive basement area as absence.
        return frame["BsmtQual"].isna() & (frame["TotalBsmtSF"].isna() | frame["TotalBsmtSF"].eq(0))

    @staticmethod
    def _no_garage(frame: pd.DataFrame) -> pd.Series:
        return frame["GarageType"].isna() & frame["GarageArea"].eq(0)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "numeric_medians_"):
            raise RuntimeError("Call fit(train) before transform(data).")
        frame = data.copy()
        no_garage = self._no_garage(frame)
        no_basement = self._no_basement(frame)

        # Explicit absence fields, verified using their physical companion measures.
        for column, label in ABSENCE_LABELS.items():
            if column in {"GarageType", "GarageFinish", "GarageQual", "GarageCond"}:
                frame.loc[frame[column].isna() & no_garage, column] = label
                frame[column] = frame[column].fillna("Missing")
            elif column == "PoolQC":
                absent = frame["PoolArea"].eq(0)
                frame.loc[frame[column].isna() & absent, column] = label
                frame[column] = frame[column].fillna("Missing")
            else:
                frame[column] = frame[column].fillna(label)

        for column in BASEMENT_CATEGORICAL:
            frame.loc[frame[column].isna() & no_basement, column] = "NoBasement"
            frame[column] = frame[column].fillna("Missing")
        for column in BASEMENT_NUMERIC:
            frame.loc[frame[column].isna() & no_basement, column] = 0
            self._fill_by_group(frame, column, "Neighborhood", {}, self.numeric_medians_[column])

        # Blank type is absence only when the area establishes no veneer.
        no_veneer = frame["MasVnrType"].isna() & frame["MasVnrArea"].eq(0)
        frame.loc[no_veneer, "MasVnrType"] = "NoMasonryVeneer"
        frame["MasVnrType"] = frame["MasVnrType"].fillna("Missing")
        self._fill_by_group(
            frame, "MasVnrArea", "Neighborhood", self.neighborhood_medians_["MasVnrArea"], self.numeric_medians_["MasVnrArea"]
        )
        self._fill_by_group(
            frame, "LotFrontage", "Neighborhood", self.neighborhood_medians_["LotFrontage"], self.numeric_medians_["LotFrontage"]
        )

        # Structural garages use zero for the year and measures only when absent.
        for column in ("GarageYrBlt", "GarageCars", "GarageArea"):
            frame.loc[frame[column].isna() & no_garage, column] = 0
            self._fill_by_group(
                frame, column, "GarageType", self.garage_type_medians_[column], self.numeric_medians_[column]
            )

        for column in GENERIC_CATEGORICAL:
            frame[column] = frame[column].fillna("Missing")
        # The data dictionary specifically says to assume typical functionality.
        frame["Functional"] = frame["Functional"].fillna("Typ")

        unresolved = frame.columns[frame.isna().any()].tolist()
        if unresolved:
            raise ValueError(f"Unresolved missing values: {unresolved}")
        return frame


def main() -> None:
    train = pd.read_csv(INPUT_TRAIN)
    test = pd.read_csv(INPUT_TEST)
    processor = MissingDataPreprocessor().fit(train)
    processed_train = processor.transform(train)
    processed_test = processor.transform(test)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    processed_train.to_csv(OUTPUT_DIR / "train_missing_processed.csv", index=False)
    processed_test.to_csv(OUTPUT_DIR / "test_missing_processed.csv", index=False)
    print(f"Wrote {OUTPUT_DIR / 'train_missing_processed.csv'} ({processed_train.shape[0]} rows)")
    print(f"Wrote {OUTPUT_DIR / 'test_missing_processed.csv'} ({processed_test.shape[0]} rows)")


if __name__ == "__main__":
    main()

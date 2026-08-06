"""
Regenerates artifacts/categories.json with the FULL set of category values
for rest_type, dish_liked, cuisines, and listed_in(type) -- exactly reproducing
the cleaning steps used in DropNa_EDA_Models.ipynb (Cells 2, 13, 14, 15, 17)
so the dropdown values line up with what the OneHotEncoder saw at fit time.

Run this once locally against the raw zomato.csv:

    python src/extract_categories.py --csv /path/to/zomato.csv

It does NOT retrain anything -- it only rebuilds the category lists needed
for the Streamlit dropdowns. Retraining still happens in the notebook.
"""
import argparse
import json
import numpy as np
import pandas as pd


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[
        "url", "address", "phone", "location",
        "reviews_list", "menu_item", "listed_in(city)"
    ], errors="ignore")
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    df["rest_type"] = df["rest_type"].astype(str).str.title().str.strip()
    df["rest_type"] = df["rest_type"].replace("Nan", np.nan)
    df["rest_type"] = df["rest_type"].fillna(df["rest_type"].mode()[0])

    df["dish_liked"] = df["dish_liked"].astype(str).str.strip().str.title()
    df["dish_liked"] = df["dish_liked"].replace(["", "Nan", "None"], np.nan)

    df["cuisines"] = df["cuisines"].astype(str).str.strip().str.title()
    df["cuisines"] = df["cuisines"].replace(["", "Nan", "None", "Null", "nan", "null"], np.nan)
    df["cuisines"] = df["cuisines"].fillna(df["cuisines"].mode()[0])

    df["listed_in(type)"] = df["listed_in(type)"].astype(str).str.strip().str.title()

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to raw zomato.csv")
    ap.add_argument("--out", default="artifacts/categories.json")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = clean(df)

    categories = {
        "rest_type": sorted(df["rest_type"].dropna().unique().tolist()),
        "dish_liked": sorted(df["dish_liked"].dropna().unique().tolist()),
        "cuisines": sorted(df["cuisines"].dropna().unique().tolist()),
        "listed_in(type)": sorted(df["listed_in(type)"].dropna().unique().tolist()),
        "votes": {"min": int(df["votes"].min()) if "votes" in df else None,
                  "max": int(df["votes"].max()) if "votes" in df else None},
    }

    with open(args.out, "w") as f:
        json.dump(categories, f, indent=2)

    print(f"Wrote {args.out}")
    for k in ("rest_type", "dish_liked", "cuisines", "listed_in(type)"):
        print(f"  {k}: {len(categories[k])} unique values")


if __name__ == "__main__":
    main()

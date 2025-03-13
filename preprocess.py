from hypothesis import given, strategies as st, assume
import pandas as pd
from datasets import load_dataset

# ds = load_dataset("open-llm-leaderboard/contents")
# df = pd.read_parquet("hf://datasets/open-llm-leaderboard/contents/data/train-00000-of-00001.parquet")
df = pd.read_json('data_4072.json', orient='records')
@given(st.just(df))  # Use `st.just()` to test the existing DataFrame
def test_existing_dataframe_null_values(df):
    """Check if the existing DataFrame contains None, NaN, or null values."""

    assume(df is not None)  # Ensure df exists
    assert not df.isna().any().any(), "The DataFrame should contain at least one None/NaN value"

test_existing_dataframe_null_values()
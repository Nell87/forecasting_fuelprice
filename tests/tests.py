import pandas as pd

def analysis_df():
    analysis_df = pd.DataFrame(
        {
            "Date": ["2022-01-01", "2022-01-02"] * 4,
            "Price": [1,2] * 4
        }
    )
    return analysis_df
  
 def test_df_column():

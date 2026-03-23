import pandas as pd
import os

class ScienceQALocalLoader:
    def __init__(self, file_path, subset_size=100):
        self.file_path = file_path
        self.subset_size = subset_size
        self.choices_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
        self.df = pd.read_parquet(self.file_path)

    def preprocess_for_r3_quant(self):
        reasoning_col = 'solution' if 'solution' in self.df.columns else 'lecture'
        mask = (self.df[reasoning_col].notnull()) & \
               (self.df[reasoning_col].str.len() > 0) & \
               (self.df['image'].notnull())
        filtered_df = self.df[mask].copy()
        filtered_df = filtered_df.rename(columns={reasoning_col: 'reasoning'})
        return filtered_df.head(self.subset_size)

    @staticmethod
    def robust_science_qa_matcher(pred, target_letter):
        pred = str(pred).strip().upper()
        patterns = [f"{target_letter}.", f"({target_letter})", f" {target_letter} "]
        if any(p in f" {pred} " for p in patterns) or (len(pred) > 0 and pred[0] == target_letter):
            return 1.0
        return 0.0

    # if __name__ == "__main__":
    #     DATA_PATH = r"./data/science_qa/validation-00000-of-00001-6c7328ff6c84284c.parquet"
    #     loader = ScienceQALocalLoader(DATA_PATH, subset_size=5)
    #     test_subset = loader.preprocess_for_r3_quant()
    #     for idx, row in test_subset.iterrows():
    #         ans_letter = loader.choices_map[int(row['answer'])]
    #         print(f"ID: {idx} | Target: {ans_letter} | CoT: {row['reasoning'][:50]}...")
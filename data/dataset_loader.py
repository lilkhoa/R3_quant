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

class DocumentVQALocalLoader:
    """
    Loader for the HuggingFaceM4/DocumentVQA dataset stored as a local parquet file.

    Dataset schema (HuggingFaceM4/DocumentVQA):
        questionId          - int32, unique question identifier
        question            - string, the question text
        question_types      - list[string], question category tags
        image               - Image, document page image
        docId               - int32, document identifier
        ucsf_document_id    - string
        ucsf_document_page_no - string
        answers             - list[string], one or more accepted answer strings

    Primary use-case: producing calibration strings for GPTQ quantization,
    analogous to ScienceQALocalLoader.get_calibration_data().
    """

    def __init__(self, file_path: str, subset_size: int = 8):
        self.file_path = file_path
        self.subset_size = subset_size
        self.df = pd.read_parquet(self.file_path)

    def preprocess_for_calibration(self) -> pd.DataFrame:
        """
        Filter rows that have both a question and at least one answer,
        then return up to subset_size rows with standardised columns:
            question  - str
            answer    - str  (first accepted answer)
            image     - raw image value from the parquet file
        """
        import numpy as np

        df = self.df.copy()

        # Keep rows where question is non-empty
        mask = df["question"].notnull() & (df["question"].str.strip().str.len() > 0)

        # Keep rows where answers is a non-empty list/array
        # Parquet stores list columns as numpy arrays, so check both.
        def _has_answers(a):
            if isinstance(a, (list, tuple)):
                return len(a) > 0
            if isinstance(a, np.ndarray):
                return a.size > 0
            return False

        mask &= df["answers"].apply(_has_answers)

        df = df[mask].copy()

        if df.empty:
            raise ValueError(
                f"[DocumentVQALocalLoader] No valid rows found in '{self.file_path}'. "
                "Check that the file exists and has non-empty 'question' and 'answers' columns."
            )

        # Normalise: use the first accepted answer as the canonical answer
        df["answer"] = df["answers"].apply(
            lambda a: str(a[0]) if (isinstance(a, (list, tuple, np.ndarray)) and len(a) > 0) else ""
        )

        return df[["question", "answer", "image"]].head(self.subset_size).reset_index(drop=True)

    def get_calibration_strings(self) -> list:
        """
        Return a list of plain-text strings suitable for GPTQ calibration.
        Format mirrors ScienceQALocalLoader: "Question: ...\nAnswer: ..."
        """
        df = self.preprocess_for_calibration()
        return [
            f"Question: {row['question']}\nAnswer: {row['answer']}"
            for _, row in df.iterrows()
        ]

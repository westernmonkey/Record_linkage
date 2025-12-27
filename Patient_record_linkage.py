import pandas as pd
from recordlinkage.datasets import load_febrl4
from splink import Linker, DuckDBAPI, block_on
import splink.comparison_library as cl
import splink.comparison_level_library as cll

dfA, dfB , links = load_febrl4(return_links = True)

dfA['source_dataset'] = "Apollo Hospitals"
dfB['source_dataset'] = "Narayana Healthcare"

dfA = dfA.reset_index().rename(columns={"rec_id": "unique_id"})
dfB = dfB.reset_index().rename(columns={"rec_id": "unique_id"})


df_combined = pd.concat([dfA, dfB], ignore_index=True)
links.to_frame().to_csv("hidden_answer_key.csv")

df_combined.to_csv("input_data.csv", index=False)

print(df_combined)

db_api = DuckDBAPI()

blocking_rules = [
    block_on("given_name"),
    block_on("surname"),
    block_on("date_of_birth")
]

settings = {
    "link_type": "link_only",
    "unique_id_column_name": "unique_id",
    "blocking_rules_to_generate_predictions": blocking_rules,
    "comparisons": [
        cl.ExactMatch("given_name"),
        cl.ExactMatch("surname"),
        cl.ExactMatch("date_of_birth"),
        cl.ExactMatch("street_number"),
        cl.ExactMatch("postcode")
    ]
}

linker = Linker(df_combined, settings, db_api)

print("Linker Initialized successfully!")

linker.training.estimate_u_using_random_sampling(max_pairs=1e6)
training_rule = block_on("given_name")
linker.training.estimate_parameters_using_expectation_maximisation(training_rule)

training_rule_2 = block_on("surname")
linker.training.estimate_parameters_using_expectation_maximisation(training_rule_2)

df_predictions = linker.inference.predict(threshold_match_probability=0.5)

df_predictions.as_pandas_dataframe().to_csv("final_predictions.csv", index=False)

print("Predictions complete!")
print("Preview of matches found:")
print(df_predictions.as_pandas_dataframe())

ground_truth = pd.read_csv("hidden_answer_key.csv")
print("The columns in your Answer Key are:")
print(list(ground_truth.columns))


col_1 = ground_truth.columns[0]
col_2 = ground_truth.columns[1]

print(f"\nMerging using found columns: '{col_1}' and '{col_2}'")

merged = pd.merge(
    df_predictions.as_pandas_dataframe(),
    ground_truth,
    left_on=["unique_id_l", "unique_id_r"],
    right_on=[col_1, col_2] 
)

# 5. REPORT CARD
true_positives = len(merged)
total_predictions = len(df_predictions.as_pandas_dataframe())
total_actual_matches = 5000

print(f"\n--- REPORT CARD ---")
print(f"Total Matches Found:     {total_predictions}")
print(f"Correct Matches:         {true_positives}")
print(f"Wrong Matches (False Pos): {total_predictions - true_positives}")
print(f"Missed Matches:          {total_actual_matches - true_positives}")
print(f"-----------------------")
print(f"Precision (Trust): {true_positives / total_predictions:.2%}")
print(f"Recall (Coverage): {true_positives / total_actual_matches:.2%}")

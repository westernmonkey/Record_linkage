import pandas as pd
from recordlinkage.datasets import load_febrl4
dfA, dfB , links = load_febrl4(return_links = True)

dfA['source_dataset'] = "Apollo Hospitals"
dfB['source_dataset'] = "Narayana Healthcare"

dfA = dfA.reset_index().rename(columns={"rec_id": "unique_id"})
dfB = dfB.reset_index().rename(columns={"rec_id": "unique_id"})


df_combined = pd.concat([dfA, dfB], ignore_index=True)
links.to_frame().to_csv("hidden_answer_key.csv")

df_combined.to_csv("input_data.csv", index=False)

print(df_combined)


# 1. Import the Generic Linker and the DuckDB Backend
from splink import Linker, DuckDBAPI, block_on

# 2. Import the Helper Libraries (for logic)
import splink.comparison_library as cl
import splink.comparison_level_library as cll

print("Success! Splink 4 is loaded.")


# 1. INITIALIZE THE ENGINE (New in Version 4)
# This creates the SQL brain that runs inside Python.
db_api = DuckDBAPI()

# 2. DEFINE BLOCKING RULES (The Filter)
# We use the 'block_on' helper to make this easy.
# These rules say: "Only look at pairs where the First Name is identical OR Surname is identical..."
blocking_rules = [
    block_on("given_name"),
    block_on("surname"),
    block_on("date_of_birth")
]

# 3. DEFINE SETTINGS (The Handbook)
settings = {
    "link_type": "link_only",
    "unique_id_column_name": "unique_id",
    "blocking_rules_to_generate_predictions": blocking_rules,
    "comparisons": [
        # We need at least basic comparisons to start
        cl.ExactMatch("given_name"),
        cl.ExactMatch("surname"),
        cl.ExactMatch("date_of_birth"),
        cl.ExactMatch("street_number"),
        cl.ExactMatch("postcode")
    ]
}

# 4. INITIALIZE THE LINKER
# Notice the change: We pass 'db_api' as the 3rd argument.
linker = Linker(df_combined, settings, db_api)

print("Linker Initialized successfully!")

# 1. TEACH "U" (Random Chance)
# We pick 1 million random pairs to learn how common "Smith" vs "Xavier" is.
linker.training.estimate_u_using_random_sampling(max_pairs=1e6)

print("------------------------------------------------")
print("U-Probabilities Learned (Random Chance)")
print("------------------------------------------------")

# 2. TEACH "M" (Typo Rates)
# We use Expectation Maximization (EM).
# Logic: "Block on First Name. If First Name matches, assume they are similar.
# Now check how often the Surname and DOB match."
training_rule = block_on("given_name")
linker.training.estimate_parameters_using_expectation_maximisation(training_rule)

print("------------------------------------------------")
print("M-Probabilities Learned (Typo Rates)")
print("------------------------------------------------")

# 3. TEACH "M" AGAIN (Round 2)
# This time, we trust the SURNAME, so the model can learn about typos in FIRST NAME.
training_rule_2 = block_on("surname")
linker.training.estimate_parameters_using_expectation_maximisation(training_rule_2)

print("------------------------------------------------")
print("Training Complete! All parameters learned.")
print("------------------------------------------------")


# 1. RUN THE PREDICTION (Updated for Splink 4)
# The function moved to the '.inference' namespace
df_predictions = linker.inference.predict(threshold_match_probability=0.5)

# 2. SAVE THE RESULTS
# df_predictions is a "SplinkDataFrame", so we convert it to Pandas first.
df_predictions.as_pandas_dataframe().to_csv("final_predictions.csv", index=False)

print("Predictions complete!")
# We can't use len() directly on a SplinkDataFrame, so we count via SQL or just display head
print("Preview of matches found:")
print(df_predictions.as_pandas_dataframe())

# 1. Load the file again
ground_truth = pd.read_csv("hidden_answer_key.csv")

# 2. PRINT THE ACTUAL COLUMN NAMES (This reveals the truth)
print("The columns in your Answer Key are:")
print(list(ground_truth.columns))

# 3. AUTOMATIC FIX ATTEMPT
# I will try to guess the column names based on common patterns.
# Usually, the first two columns are the IDs.
col_1 = ground_truth.columns[0]
col_2 = ground_truth.columns[1]

print(f"\nMerging using found columns: '{col_1}' and '{col_2}'")

# 4. GRADE THE EXAM (Using the detected names)
merged = pd.merge(
    df_predictions.as_pandas_dataframe(),
    ground_truth,
    left_on=["unique_id_l", "unique_id_r"],
    right_on=[col_1, col_2] # We use the detected names here
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
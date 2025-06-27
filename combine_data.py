import pandas as pd

# Adjust paths if your files are not in the same directory as this script
train_df = pd.read_csv('F:\\Mani\\Projects\\arrhythmia-detection\\dataset\\mitbih_train.csv', header=None)
test_df = pd.read_csv('F:\\  Mani\\Projects\\arrhythmia-detection\\dataset\\mitbih_test.csv', header=None)

# Concatenate them
combined_df = pd.concat([train_df, test_df], ignore_index=True)

# Save the combined DataFrame to a new CSV file
# IMPORTANT: Do not include header, as the original files don't have one and
# your script expects the last column to be the label by index initially.
combined_df.to_csv('combined_mitbih.csv', index=False, header=False)

print("Combined dataset saved as combined_mitbih.csv")
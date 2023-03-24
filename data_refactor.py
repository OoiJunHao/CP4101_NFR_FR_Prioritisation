"""
Since all the scripts output the full data in a csv format, it can become
very time and space consuming when running high levels of landscape_repititions*agent_number*search_iterations.

Use this script to convert the full csv files into compressed bz2 files, as well as extract out the data that is of interest.

Or, go to the script and edit the csv output portion to include a compression technique.

Edit names of files and reformatting logic as required.
"""
import pandas as pd

# Edit names to paths of interest
file_names = [
    "availability__50__together",
    "availability__70__together",
    "availability__100__together",
    "availability__50__separate",
    "availability__70__separate",
    "availability__100__separate",
]

for file_name in file_names:
    # ---read-full-name---
    df = pd.read_csv(f"{file_name}.csv", header=None)
    print(df.size)
    print(df.head())

    # ---out-compressed file---
    df.to_csv(f"{file_name}.bz2", index=False, compression="bz2")

    # ---load---
    result_read = pd.read_csv(f"{file_name}.bz2", compression="bz2")
    print(result_read.size)
    print(result_read.head())

    # --obtain last column values---
    values = result_read[result_read.columns[-1]]
    pd.DataFrame(values).to_csv(f"{file_name}_final_values.csv", header=False)


"""
Version 2
"""
# import pandas as pd

# # Edit names to paths of interest
# file_names = [
#     "availability__50__together",
#     "availability__70__together",
#     "availability__100__together",
#     "availability__50__separate",
#     "availability__70__separate",
#     "availability__100__separate",
# ]
# final_values = []

# for file_name in file_names:
#     # ---read-full-name---
#     df = pd.read_csv(f"{file_name}.csv", header=None)
#     print(df.size)
#     print(df.head())

#     # ---out-compressed file---
#     df.to_csv(f"{file_name}.bz2", index=False, compression="bz2")

#     # ---load---
#     result_read = pd.read_csv(f"{file_name}.bz2", compression="bz2")
#     print(result_read.size)
#     print(result_read.head())

#     # --obtain last column values---
#     values = result_read[result_read.columns[-1]]
#     pd.DataFrame(values).to_csv(f"{file_name}_final_values.csv", header=False)

# final_df = pd.concat(final_values, axis=1, ignore_index=True)
# final_df.to_csv("")

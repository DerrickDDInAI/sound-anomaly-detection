"""
Script to
1. read all sounds
2. extract their features
3. export the dataset to a csv file

Important note: this script currently takes a bit less than 9 hours
to build a dataset of about 54000 sounds with all their features.
"""

# =====================================================================
# Import
# =====================================================================

# Import internal modules
import os
import time
from threading import Thread, RLock
from typing import List

# Import 3rd party modules
import pandas as pd

# Import local modules
from sound import get_all_sounds, get_audio_features, audio_features_cols


# =====================================================================
# Class
# =====================================================================

# Instantiate RLock object
writing_csv_lock = RLock()

# Define child class of Thread


class SyncThread(Thread):
    """
    Child class of thread
    to read the sounds in 4 threads:
    1 thread per machine type
    """

    def __init__(self, machine_type: List[str]):
        """
        Function to create an instance of SyncThread class
        """
        Thread.__init__(self)
        self.machine_type: List[str] = machine_type

    def run(self):
        """
        Function to start the thread
        """
        # Track time to run the thread
        start_time = time.time()
        print(f"Starting thread {self.machine_type}")

        # Get sound filepaths
        df_thread = get_all_sounds([-6, 0, 6], self.machine_type)

        # Extract and add audio features to the dataframe
        df_thread[audio_features_cols] = df_thread[["sound_path"]].apply(
            lambda x: pd.Series(get_audio_features(x.sound_path)), axis=1)

        end_time = time.time()
        print(end_time - start_time)

        # Export dataframe to csv
        with writing_csv_lock:
            print(f"writing {self.machine_type} sounds to csv")

            # If file exists, don't repeat writing the header
            file_exists = os.path.isfile(os.path.join(
                "assets", "data", "thread_csv_all.csv"))
            if file_exists:
                df_thread.to_csv(os.path.join(
                    "assets", "data", "thread_csv_all.csv"), header=False, index=False, mode='a')
            else:
                df_thread.to_csv(os.path.join(
                    "assets", "data", "thread_csv_all.csv"), header=True, index=False, mode='a')


# =====================================================================
# Run
# =====================================================================

run_script: bool = True

# Check if file it exists
file_exists = os.path.isfile(os.path.join(
    "assets", "data", "thread_csv_all.csv"))
if file_exists:
    # For safety reasons, ask the user if he's sure to run the script
    delete_or_not: str = input("""
    The csv file already exists.
    This script will replace the current csv file.
    Do you still want to run it? (yes/no): 
    """).lower()
    if delete_or_not in ("yes", "y"):
        os.remove(os.path.join("assets", "data", "thread_csv_all.csv"))
    else:
        run_script = False

if run_script:
    # Create threads
    thread_1 = SyncThread(["fan"])
    thread_2 = SyncThread(["pump"])
    thread_3 = SyncThread(["slider"])
    thread_4 = SyncThread(["valve"])

    # Launch threads
    thread_1.start()
    thread_2.start()
    thread_3.start()
    thread_4.start()

    # Add threads to thread list
    threads = []
    threads.append(thread_1)
    threads.append(thread_2)
    threads.append(thread_3)
    threads.append(thread_4)

    # Wait for all threads to complete
    for t in threads:
        t.join()
    print("Exiting Main Thread")

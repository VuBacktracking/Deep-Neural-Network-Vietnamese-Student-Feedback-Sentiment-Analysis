import pandas as pd
import os

def fetch_all_data():

    new_directory_path = 'Data'

    if not os.path.exists(new_directory_path):
        # Create the directory
        os.mkdir(new_directory_path)
        print(f"Directory '{new_directory_path}' created successfully.")
    else:
        print(f"Directory '{new_directory_path}' already exists.")

    list_data_names = ["train", "test", "dev"]
    
    def fetch_data_to_csv(folder_name):
        path = "_UIT-VSFC/" + folder_name

        # Read the text files as plain text and remove newline characters
        # Sents
        with open(path + "/sents.txt", "r", encoding="utf-8") as file:
            sents = [line.replace("\n", "") for line in file.readlines()]
        # Topics
        with open(path + "/topics.txt", "r", encoding="utf-8") as file:
            topics = [line.replace("\n", "") for line in file.readlines()]
        # Sentiments
        with open(path + "/sentiments.txt", "r", encoding="utf-8") as file:
            sentiments = [line.replace("\n", "") for line in file.readlines()]

        sents_df = pd.DataFrame({'sents': sents})
        topics_df = pd.DataFrame({'topics': topics})
        sentiments_df = pd.DataFrame({'sentiments': sentiments})

        merged_df = pd.concat([sents_df, topics_df, sentiments_df], axis=1)

        merged_df.to_csv(f'Data/{folder_name}.csv', index=False)
    
    for folder_name in list_data_names:
        fetch_data_to_csv(folder_name)
    
    print("Sucessfully Fetching")
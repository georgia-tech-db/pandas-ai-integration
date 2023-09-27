import json
import csv
import os
import evadb
import pandas as pd

# Specify the directory containing your JSON files and the desired CSV file name
JSON_DIRECTORY = "./atlanta"
PROJECT_NAME = "postgres"

CSV_FILE_PATH = f'{PROJECT_NAME}.csv'

# Initialize an empty list to store the combined data from all JSON files
combined_data = []

# Iterate through each JSON file in the directory
for filename in os.listdir(JSON_DIRECTORY):
    if filename.endswith('.json'):
        json_file_path = os.path.join(JSON_DIRECTORY, filename)
        
        # Open the JSON file for reading
        with open(json_file_path, 'r', encoding='utf-8') as json_input_file:
            # Load the JSON data from the file
            json_data = json.load(json_input_file)
            for json_obj in json_data:
                json_obj['date'] =\
                os.path.basename(str(json_file_path))
            
            # Append the JSON data to the combined_data list
            combined_data.extend(json_data)

# Specify the headers for your CSV file based on the keys present in the JSON data
# This will ensure that only common keys across all JSON objects are included
csv_headers = list(set().union(*(d.keys() for d in combined_data)))

# Open the CSV file for writing
with open(CSV_FILE_PATH, 'w', newline='', encoding='utf-8') as csv_output_file:
    # Create a CSV writer
    csv_writer = csv.DictWriter(csv_output_file, fieldnames=csv_headers)
    
    # Write the headers to the CSV file
    csv_writer.writeheader()
    
    # Write the combined JSON data to the CSV file
    csv_writer.writerows(combined_data)

print(f'Conversion from JSON to CSV complete. Data saved to {CSV_FILE_PATH}')

# Specify the input CSV file and output CSV file
input_csv_file = CSV_FILE_PATH
output_csv_file = CSV_FILE_PATH

# Define the old and new column names
old_column_name = 'metadata'
new_column_name = 'metadata_slack'

# Read the input CSV file and create a list of rows
with open(input_csv_file, 'r', newline='', encoding='utf-8') as input_file:
    # Create a CSV reader
    csv_reader = csv.reader(input_file)
    
    # Read the header row
    header = next(csv_reader)
    
    # Find the index of the old column name in the header
    try:
        old_index = header.index(old_column_name)
    except ValueError:
        # Handle the case where the old column name is not found in the header
        print(f'Column name "{old_column_name}" not found in the header.')
        exit(1)

    # Update the header with the new column name
    header[old_index] = new_column_name

    # Read the rest of the rows
    rows = list(csv_reader)

# Write the modified CSV data to the output file
with open(output_csv_file, 'w', newline='', encoding='utf-8') as output_file:
    # Create a CSV writer
    csv_writer = csv.writer(output_file)
    
    # Write the updated header
    csv_writer.writerow(header)
    
    # Write the rest of the rows
    csv_writer.writerows(rows)

print(f'Column name "{old_column_name}" has been changed to "{new_column_name}" in {output_csv_file}')

if __name__ == "__main__":
    try:
        # establish evadb api cursor
        print("⏳ Establishing evadb connection...")
        cursor = evadb.connect().cursor()
        print("✅ evadb connection setup complete!")

        print(f'{CSV_FILE_PATH}')

        cursor.query(f"DROP FUNCTION IF EXISTS Chunk;").df()

        cursor.query(f"""
            CREATE FUNCTION Chunk
            INPUT (text TEXT(1000))
            OUTPUT (chunks TEXT(1000))
            TYPE  StringProcessing
            IMPL  'chunk.py';
        """).df()

        cursor.query(f"DROP FUNCTION IF EXISTS Contains;").df()

        cursor.query(f"""
            CREATE FUNCTION Contains
            INPUT (input_string TEXT(1000), substring TEXT(1000))
            OUTPUT (contains BOOLEAN)
            TYPE  StringProcessing
            IMPL  'contains.py';
        """).df()        

        cursor.query(f"DROP TABLE IF EXISTS SlackCSV;").df()

        cursor.query(f"""CREATE TABLE SlackCSV(
            blocks TEXT(1000),
            user_profile TEXT(1000),
            reply_count TEXT(1000),
            edited TEXT(1000),
            user TEXT(1000),
            username TEXT(1000),
            bot_id INTEGER,
            text TEXT(1000),
            user_team TEXT(1000),
            replies TEXT(1000),
            icons TEXT(1000),
            hidden TEXT(1000),
            delete_original TEXT(1000),
            pinned_to TEXT(1000),
            latest_reply TEXT(1000),
            old_name TEXT(1000),
            team TEXT(1000),
            reply_users TEXT(1000),
            metadata_slack TEXT(1000),
            replace_original TEXT(1000),
            subscribed TEXT(1000),
            reply_users_count TEXT(1000),
            parent_user_id TEXT(1000),
            thread_ts TEXT(1000),
            attachments TEXT(1000),
            subtype TEXT(1000),
            last_read TEXT(1000),
            client_msg_id TEXT(1000),
            bot_profile TEXT(1000),
            reactions TEXT(1000),
            files TEXT(1000), 
            name TEXT(1000),
            inviter TEXT(1000),
            upload TEXT(1000),
            type TEXT(1000),
            ts TEXT(1000),
            purpose TEXT(1000),
            source_team TEXT(1000),
            date TEXT(1000)
            );
          """).df()

        cursor.query(f"LOAD CSV '{CSV_FILE_PATH}' INTO SlackCSV;").df()

        pd.set_option('display.max_columns', None)  # Show all columns
        pd.set_option('display.expand_frame_repr', False)
        pd.set_option('display.max_colwidth', None)
        print("here1")
        # execute a select query
        select_query = cursor.query(
            """SELECT Chunk(text)
                    FROM SlackCSV
                    WHERE _row_id < 100 AND Contains(text, "predict") = "True";
            """).df()
        print("here2")
        print(select_query)

    except Exception as e:
        print("❗️ Session ended with an error.")
        print(e)

exit(0)

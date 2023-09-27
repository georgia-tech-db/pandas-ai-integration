import os
import subprocess
import sys
import openai
import evadb

# Replace 'your-api-key' with your OpenAI API key
openai.api_key = "sk-xx"

MAX_CHUNK_SIZE=15000

# Check if the search term argument is provided
if len(sys.argv) != 2:
    print("Usage: python script.py <search_term>")
    sys.exit(1)


# Extract the search term from the command line arguments
search_term = sys.argv[1]

# Define the directory where you want to search for JSON files
search_directory = "./"

# Define the output file name
output_file = "data.txt"

# Construct the find command to search for JSON files containing the specified term
find_command = f'find "{search_directory}" -name "*.json" -exec grep -Hn --color "{search_term}" {{}} \\; > "{output_file}"'

# Execute the find command
os.system(find_command)

print(f"Search results saved to {output_file}")

# Function to split text into chunks of MAX_CHUNK_SIZE characters or less, stopping at the nearest newline
def split_text_into_chunks(text, max_chunk_size=MAX_CHUNK_SIZE):
    chunks = []
    current_chunk = ""

    for line in text.splitlines():
        if len(current_chunk) + len(line) + 1 <= max_chunk_size:
            # Add line to the current chunk
            if current_chunk:
                current_chunk += '\n'
            current_chunk += line
        else:
            # Start a new chunk
            chunks.append(current_chunk)
            current_chunk = line

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

# Read the contents of the "data.txt" file
with open(output_file, 'r') as file:
    file_contents = file.read()

# Split the contents into chunks
text_chunks = split_text_into_chunks(file_contents)

# Save the text chunks to a new file
chunked_output_file = "data_chunks.txt"
i = 0
with open(chunked_output_file, 'w') as chunked_file:
    for chunk in text_chunks:
        i = i + 1
        chunk = chunk + '\n\n\n\n\n\n\n'
        chunked_file.write(chunk)
        print("chunk " + str(i))

print(f"Text chunks saved to {chunked_output_file}")

# Initialize an empty list to store responses
responses = []

# Create a function to generate responses using the chat model
def generate_chat_response(prompt):
    try:
        print("done")
        return "tmp"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",  # Use the appropriate chat model
            messages=[
                {"role": "system", "content": "You are a junior programmer trying to summarize user experience issues."},
                {"role": "user", "content": prompt},
            ],
            stop=None,  # You can specify a stop condition if necessary
            temperature=0.7,  # You can adjust the temperature for creativity
        )
        return response.choices[0].message['content']
    except Exception as e:
        return str(e)

# Question to add to the prompt
question = """Summarize the user complaints in these JSON messages. Along with each complaint, provide the relevant user messages and file names (e.g., questions/2022-06-03.json). --- """


# Iterate through each chunk and query ChatGPT with the question
for chunk in text_chunks:
    prompt = f"{question}\n{chunk}"  # Add the question to the chunk
    response = generate_chat_response(prompt)
    print(response)
    responses.append(response)

# Save the responses to a new file
responses_output_file = "responses.txt"
with open(responses_output_file, 'w') as responses_file:
    for response in responses:
        responses_file.write(response + '\n')

print(f"Responses saved to {responses_output_file}")


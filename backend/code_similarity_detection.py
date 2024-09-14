import os
import re
import ast
import io
import zipfile
import tempfile
import tokenize
from difflib import SequenceMatcher
from simhash import Simhash
from backend.normalizedAST import ASTNormalizer

# Function to tokenize code
def tokenize_code(code):
    return re.findall(r'\b\w+\b', code)  # Use regex to find word-like tokens in the code

# Function to generate hash signature using Simhash algorithm
def generate_hash_signature(tokens):
    return Simhash(tokens).value  # Generate and return the Simhash value for the tokens

# Function to calculate text similarity using hamming distance
def calculate_text_similarity(code1, code2):
    tokens1 = tokenize_code(code1)  # Tokenize the first code snippet
    tokens2 = tokenize_code(code2)  # Tokenize the second code snippet
    
    hash_signature1 = generate_hash_signature(tokens1)  # Generate hash signature for the first code
    hash_signature2 = generate_hash_signature(tokens2)  # Generate hash signature for the second code
    
    # Calculate the Hamming distance between the two hash signatures
    distance = bin(hash_signature1 ^ hash_signature2).count('1')
    
    # Calculate similarity as 1 minus the normalized Hamming distance
    similarity = 1 - (distance / 64)  # Simhash uses 64-bit hash by default
    return similarity  # Return the calculated similarity

# Function to parse code to AST and normalize it
def parse_and_normalize_code(code):
    try:
        tree = ast.parse(code)
        normalizer = ASTNormalizer()
        return normalizer.visit(tree)
    except SyntaxError:
        return None

# Function to compare ASTs
def compare_asts(ast1, ast2):
    if ast1 is None or ast2 is None:  # Check if either AST is None
        return 0  # Return 0 if either AST is None
    ast_str1 = ast.dump(ast1)  # Convert the first AST to a string representation
    ast_str2 = ast.dump(ast2)  # Convert the second AST to a string representation
    similarity_ratio = SequenceMatcher(None, ast_str1, ast_str2).ratio()  # Calculate similarity using SequenceMatcher
    return similarity_ratio  # Return the similarity ratio

# Function to calculate structural similarity using AST comparison
def calculate_structural_similarity(code1, code2):
    ast1 = parse_and_normalize_code(code1)
    ast2 = parse_and_normalize_code(code2)
    similarity = compare_asts(ast1, ast2)
    return similarity

# Function to calculate weighted average of text and structural similarity
def calculate_weighted_similarity(text_similarity, structural_similarity):
    # Determine weights based on the text similarity value
    if text_similarity > structural_similarity and text_similarity > 0.6:
        text_weight = 0.8
        structural_weight = 0.2
    else:
        text_weight = 0.2
        structural_weight = 0.8

    # Calculate the weighted similarity score
    return (text_similarity * text_weight) + (structural_similarity * structural_weight)

# Function to format code
def format_code(code):
    io_obj = io.StringIO(code)  # Create a StringIO object from the code string
    out = []  # Initialize an empty list to store formatted code
    prev_toktype = None  # Initialize the previous token type as None
    last_lineno = -1  # Initialize the last line number
    last_col = 0  # Initialize the last column number
    
    try:
        # Iterate over tokens generated by tokenize
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type, ttext, (slineno, scol), (elineno, ecol), ltext = tok  # Unpack token properties

            if token_type == tokenize.COMMENT:  # Skip comments
                continue
            if token_type == tokenize.STRING:  # Skip docstrings
                if prev_toktype == tokenize.INDENT:
                    continue
            if slineno > last_lineno:  # Add new line if current line number is greater than the last
                last_col = 0
            if scol > last_col:  # Add spaces if the current column is greater than the last
                out.append(" " * (scol - last_col))
            out.append(ttext)  # Append token text to output list
            prev_toktype = token_type  # Update previous token type
            last_col = ecol  # Update last column number
            last_lineno = elineno  # Update last line number
    except (tokenize.TokenError, IndentationError) as e:
        print(f"Error tokenizing code: {e}")  # Print an error message if tokenizing fails
        return code  # Return the original code if there's an error

    formatted_code = "".join(out)  # Join the formatted code into a single string
    # Remove extra blank lines and spaces
    formatted_code = os.linesep.join([s for s in formatted_code.splitlines() if s.strip()])  # Remove extra blank lines
    return formatted_code  # Return the formatted code

# Function to extract files from upload
def extract_files(uploaded_files):
    temp_dir = tempfile.mkdtemp()  # Create a temporary directory
    extracted_files = []  # Initialize an empty list to store extracted file paths
    extracted_files_content = {}  # Initialize an empty dictionary to store file contents

    # Iterate over each uploaded file
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith('.zip'):  # Check if the file is a ZIP archive
            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:  # Open the ZIP file
                zip_ref.extractall(temp_dir)  # Extract all files in the ZIP archive to the temporary directory
                # Walk through the extracted files
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        if file.endswith('.py'):  # Check if the file is a Python file
                            file_path = os.path.join(root, file)  # Get the full file path
                            with open(file_path, 'r') as f:
                                content = f.read()  # Read the file content
                                extracted_files_content[file_path] = content  # Store the content in the dictionary
                            extracted_files.append(file_path)  # Add the file path to the list
        else:
            temp_path = os.path.join(temp_dir, uploaded_file.name)  # Create a path for the non-ZIP file
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())  # Write the uploaded file content to the path
            if temp_path.endswith('.py'):  # Check if the file is a Python file
                with open(temp_path, 'r') as f:
                    content = f.read()  # Read the file content
                    extracted_files_content[temp_path] = content  # Store the content in the dictionary
                extracted_files.append(temp_path)  # Add the file path to the list

    return extracted_files, extracted_files_content  # Return the extracted file paths and contents

# Function to compare files and calculate similarity
def compare_files(file_pair, extracted_files_content):
    code1_file, code2_file = file_pair  # Unpack the file pair

    try:
        # Get the contents of the two files from the dictionary
        code1 = extracted_files_content.get(code1_file, '')
        code2 = extracted_files_content.get(code2_file, '')

        formatted_code1 = format_code(code1)  # Format the first file content
        formatted_code2 = format_code(code2)  # Format the second file content

        text_similarity = calculate_text_similarity(formatted_code1, formatted_code2)  # Calculate text similarity
        structural_similarity = calculate_structural_similarity(formatted_code1, formatted_code2)  # Calculate structural similarity

        weighted_similarity = calculate_weighted_similarity(text_similarity, structural_similarity)  # Calculate weighted similarity

        # Return the filenames and calculated similarities
        return os.path.basename(code1_file), os.path.basename(code2_file), text_similarity, structural_similarity, weighted_similarity

    except IOError:
        # Return None values if there is an IOError
        return None, None, None, None, None

# Function to sanitize user input for activity title
def sanitize_title(title):
    return re.sub(r'[^A-Za-z0-9_ ]+', '', title)  # Use regex to remove invalid characters from the title

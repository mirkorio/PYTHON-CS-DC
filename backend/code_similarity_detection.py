import os
import re
import ast
import io
import zipfile
import tempfile
import tokenize
from difflib import SequenceMatcher
from simhash import Simhash
# Function to tokenize code
def tokenize_code(code):
    return re.findall(r'\b\w+\b', code)

# Function to generate hash signature using Simhash algorithm
def generate_hash_signature(tokens):
    return Simhash(tokens).value

# Function to calculate text similarity using hamming distance
def calculate_text_similarity(code1, code2):
    tokens1 = tokenize_code(code1)
    tokens2 = tokenize_code(code2)
    
    hash_signature1 = generate_hash_signature(tokens1)
    hash_signature2 = generate_hash_signature(tokens2)
    
    distance = bin(hash_signature1 ^ hash_signature2).count('1')
    similarity = 1 - (distance / 64)  # Simhash uses 64-bit hash by default
    return similarity

# Function to parse code to AST
def parse_code_to_ast(code):
    try:
        return ast.parse(code)
    except SyntaxError:
        return None

# Function to compare ASTs
def compare_asts(ast1, ast2):
    if ast1 is None or ast2 is None:
        return 0
    ast_str1 = ast.dump(ast1)
    ast_str2 = ast.dump(ast2)
    similarity_ratio = SequenceMatcher(None, ast_str1, ast_str2).ratio()
    return similarity_ratio

# Function to calculate structural similarity using AST comparison
def calculate_structural_similarity(code1, code2):
    ast1 = parse_code_to_ast(code1)
    ast2 = parse_code_to_ast(code2)
    similarity = compare_asts(ast1, ast2)
    return similarity

# Function to calculate weighted average of text and structural similarity
def calculate_weighted_similarity(text_similarity, structural_similarity):
    if text_similarity > structural_similarity and text_similarity > 0.6:
        text_weight = 0.8
        structural_weight = 0.2
    else:
        text_weight = 0.2
        structural_weight = 0.8
    return (text_similarity * text_weight) + (structural_similarity * structural_weight)

# Function to format code
def format_code(code):
    io_obj = io.StringIO(code)
    out = []
    prev_toktype = None
    last_lineno = -1
    last_col = 0
    
    try:
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type, ttext, (slineno, scol), (elineno, ecol), ltext = tok
            if token_type == tokenize.COMMENT:
                continue
            if token_type == tokenize.STRING:
                if prev_toktype == tokenize.INDENT:
                    continue
            if slineno > last_lineno:
                last_col = 0
            if scol > last_col:
                out.append(" " * (scol - last_col))
            out.append(ttext)
            prev_toktype = token_type
            last_col = ecol
            last_lineno = elineno
    except (tokenize.TokenError, IndentationError) as e:
        print(f"Error tokenizing code: {e}")
        return code

    formatted_code = "".join(out)
    # Remove extra blank lines and spaces
    formatted_code = os.linesep.join([s for s in formatted_code.splitlines() if s.strip()])
    return formatted_code

# Function to extract files from upload
def extract_files(uploaded_files):
    temp_dir = tempfile.mkdtemp()
    extracted_files = []
    extracted_files_content = {}

    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith('.zip'):
            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        if file.endswith('.py'):
                            file_path = os.path.join(root, file)
                            with open(file_path, 'r') as f:
                                content = f.read()
                                extracted_files_content[file_path] = content
                            extracted_files.append(file_path)
        else:
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            if temp_path.endswith('.py'):
                with open(temp_path, 'r') as f:
                    content = f.read()
                    extracted_files_content[temp_path] = content
                extracted_files.append(temp_path)

    return extracted_files, extracted_files_content

# Function to compare files and calculate similarity
def compare_files(file_pair, extracted_files_content):
    code1_file, code2_file = file_pair

    try:
        code1 = extracted_files_content.get(code1_file, '')
        code2 = extracted_files_content.get(code2_file, '')

        formatted_code1 = format_code(code1)
        formatted_code2 = format_code(code2)

        text_similarity = calculate_text_similarity(formatted_code1, formatted_code2)
        structural_similarity = calculate_structural_similarity(formatted_code1, formatted_code2)

        weighted_similarity = calculate_weighted_similarity(text_similarity, structural_similarity)

        return os.path.basename(code1_file), os.path.basename(code2_file), text_similarity, structural_similarity, weighted_similarity

    except IOError:
        return None, None, None, None, None

# Function to sanitize user input for activity title
def sanitize_title(title):
    return re.sub(r'[^A-Za-z0-9_ ]+', '', title)
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import re
import copy

def clean_text_columns(df):
    """
    Cleans text columns by removing common prefixes or suffixes.
    A column is considered a text column if >80% of its values are non-numeric.

    Args:
        df (pd.DataFrame): The DataFrame to process.

    Returns:
        pd.DataFrame: A new DataFrame with cleaned text columns.
    """
    df_cleaned = df.copy()
    for col in df_cleaned.columns:
        # Process only object-dtype columns (potential text)
        if df_cleaned[col].dtype == 'object':
            series = df_cleaned[col].dropna().astype(str)
            if series.empty:
                continue

            # --- Logic to determine if it's a text column ---
            # Count how many values are NOT numeric
            non_numeric_count = pd.to_numeric(series, errors='coerce').isna().sum()
            total_count = len(series)

            # If over 80% of the data is non-numeric, treat it as a text column
            if total_count > 0 and (non_numeric_count / total_count) > 0.8:
                # Find the longest common prefix among all entries
                common_prefix = os.path.commonprefix(series.tolist())
                if len(common_prefix) > 1:
                    df_cleaned[col] = df_cleaned[col].str.removeprefix(common_prefix).str.strip()
                    series = df_cleaned[col].dropna().astype(str)

                # Find the longest common suffix (by reversing strings)
                reversed_series = series.str[::-1]
                common_suffix_rev = os.path.commonprefix(reversed_series.tolist())
                common_suffix = common_suffix_rev[::-1]

                if len(common_suffix) > 1:
                    df_cleaned[col] = df_cleaned[col].str.removesuffix(common_suffix).str.strip()
    return df_cleaned


def convert_table_types(df: pd.DataFrame) -> pd.DataFrame:
    df_processed = df.copy()
    
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':
            series = df_processed[col].dropna().astype(str)
            if series.empty:
                continue

            # 1) Strip non-digits
            cleaned = series.str.replace(r'\D', '', regex=True)

            # 2) Convert to numeric, coercing errors to NaN
            numeric = pd.to_numeric(cleaned, errors='coerce')

            # If *any* value converted successfully, we treat the column as numeric
            if not numeric.isna().all():
                # 3) Check if all non-null values are integers
                non_null = numeric.dropna()
                if (non_null % 1 == 0).all():
                    df_processed[col] = numeric.astype('float64')
                else:
                    df_processed[col] = numeric

    return df_processed

def process_tables_with_custom_logic(soup):
    """
    Finds all tables in a BeautifulSoup object and processes them with custom logic
    for handling hyperlinks based on column type.

    Args:
        soup (BeautifulSoup): The parsed HTML of the page.

    Returns:
        list: A list of processed pandas DataFrames.
    """
    processed_tables = []
    table_elements = soup.find_all('table')
    print(f"Found {len(table_elements)} table element(s). Starting custom processing...")

    for i, table_soup in enumerate(table_elements):
        # Use a copy for analysis to avoid altering the original soup needed for processing
        analysis_soup = copy.copy(table_soup)

        # 1. Dry run to identify column types
        try:
            # Use flavor='lxml' for better parsing
            temp_df = pd.read_html(str(analysis_soup), flavor='lxml')[0]
        except Exception:
            print(f"Skipping table {i+1} as it could not be parsed for analysis.")
            continue

        
        numeric_coun=0
        numeric_col_indices = []
        for col_idx, col_name in enumerate(temp_df.columns):
            series = temp_df[col_name].dropna().astype(str)
            if series.empty:
                continue

            numeric_count=0
            for indx in series.index :
              val=series.loc[indx]
              cleaned = re.sub(r'[\$,£,₹,€,]', '', val)
              digits = re.findall(r'\d', cleaned)
              letters = re.findall(r'[A-Za-z]', cleaned)

              if len(digits) > len(letters) :
                numeric_count+=1
                number_only = re.sub(r'\D', '', cleaned)
                series.loc[indx]=number_only

            
            total_count = len(series)
            # Check if atleast over 10% of the column can be converted to a number
            
            if total_count > 0 and (numeric_count / total_count) > 0.7:
                # --- NEW LOGIC: Check if the numeric column is fully hyperlinked ---
                is_fully_linked = False
                rows = table_soup.find_all('tr')
                # Try to skip header row for more accurate check
                data_rows = rows[1:] if rows and rows[0].find_all('th') else rows

                non_empty_cell_count = 0
                linked_cell_count = 0
                for row in data_rows:
                    cells = row.find_all(['td', 'th'])
                    if col_idx < len(cells):
                        cell = cells[col_idx]
                        if cell.get_text(strip=True):  # Is the cell not empty?
                            non_empty_cell_count += 1
                            if cell.find('a'):  # Does it contain a link?
                                linked_cell_count += 1

                if non_empty_cell_count > 0 and non_empty_cell_count == linked_cell_count:
                    is_fully_linked = True

                # Only add to the stripping list if it's NOT fully hyperlinked
                if not is_fully_linked:
                    numeric_col_indices.append(col_idx)
                else:
                    print(f"Table {i+1}, Column {col_idx}: Identified as numeric but is fully hyperlinked. Links will be preserved.")

        print(f"Table {i+1}: Identified numeric columns for link stripping at indices: {numeric_col_indices}")

        # 2. Process the actual table soup based on column types
        for row in table_soup.find_all('tr'):
            cells = row.find_all(['td', 'th'])
            for cell_idx, cell in enumerate(cells):
                links = cell.find_all('a')
                if not links:
                    continue

                if cell_idx in numeric_col_indices:
                    # For numeric columns, remove the link and its text
                    for link in links:
                        link.decompose()
                else:
                    # For non-numeric columns, keep the link's text
                    for link in links:
                        link.unwrap()

        # 3. Convert the fully processed soup to a DataFrame
        try:
            final_df = pd.read_html(str(table_soup), flavor='lxml')[0]
            processed_tables.append(final_df)
        except Exception:
            print(f"Skipping table {i+1} as it could not be parsed after final processing.")
            continue

    return processed_tables


def scrape_and_extract(url):
    """
    Fetches a web page, extracts all text content, and finds all tables using custom logic.

    Args:
        url (str): The URL of the website to scrape.

    Returns:
        tuple: A tuple containing (all_text, list_of_tables)
    """
    print(f"Attempting to scrape: {url}")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

    except requests.exceptions.RequestException as e:
        print(f"Error: Could not retrieve the webpage. {e}")
        return None, None

    soup = BeautifulSoup(response.content, 'html.parser')
    for script_or_style in soup(['script', 'style']):
        script_or_style.decompose()
    text_content = soup.get_text(separator='\n', strip=True)

    # --- Use the new custom table processing function ---
    tables = process_tables_with_custom_logic(soup)

    # --- Post-process the extracted tables ---
    cleaned_tables = [df for df in tables]
    final_tables = [(df) for df in cleaned_tables]

    return text_content, final_tables

def save_content(url, text_content, tables):
    """
    Saves the extracted text to a .txt file and tables to .csv files.
    """
    base_filename = url.split('//')[-1].replace('/', '_').replace('.', '_')
    if len(base_filename) > 50:
        base_filename = base_filename[:50]

    text_filename = f"{base_filename}_content.txt"
    with open(text_filename, 'w', encoding='utf-8') as f:
        f.write(text_content)
    print(f"Successfully saved text content to: {os.path.abspath(text_filename)}")

    if not tables:
        print("No tables to save.")
        return

    for i, table_df in enumerate(tables):
        csv_filename = f"{base_filename}_table_{i+1}.csv"
        if not table_df.empty:
            table_df.to_csv(csv_filename, index=False, encoding='utf-8')
            print(f"Successfully saved table {i+1} to: {os.path.abspath(csv_filename)}")
        else:
            print(f"Skipping empty table {i+1}.")
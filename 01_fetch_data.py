# 01_fetch_data.py

import requests
import xml.etree.ElementTree as ET
import pandas as pd
import time

# Function to safely extract text from XML, handling missing tags
def get_text_from_element(element, tag, namespace):
    found = element.find(tag, namespace)
    return found.text.strip() if found is not None else None

def fetch_papers_for_category(category, max_results=350):
    """
    Manually fetches papers for a single category by directly calling the arXiv API.
    """
    print(f"\nSearching for {max_results} papers in '{category}'...")
    
    base_url = 'http://export.arxiv.org/api/query?'
    papers = []
    start = 0
    page_size = 100 # arXiv recommends fetching in pages of 100-200

    # Define the XML namespace to correctly parse the Atom feed
    namespace = {'atom': 'http://www.w3.org/2005/Atom'}

    while len(papers) < max_results:
        # Construct the URL for the current page of results
        query = f'search_query=cat:{category}&sortBy=submittedDate&sortOrder=descending&start={start}&max_results={page_size}'
        
        try:
            # Make the HTTP request to the API
            response = requests.get(base_url + query)
            response.raise_for_status() # Raises an exception for bad status codes (4xx or 5xx)

            # Parse the XML response
            root = ET.fromstring(response.content)
            entries = root.findall('atom:entry', namespace)

            # If a page is empty, we've reached the end of the results
            if not entries:
                print("Found an empty page. Ending search for this category.")
                break

            for entry in entries:
                papers.append({
                    'id': get_text_from_element(entry, 'atom:id', namespace),
                    'title': get_text_from_element(entry, 'atom:title', namespace),
                    'summary': get_text_from_element(entry, 'atom:summary', namespace),
                    'published': get_text_from_element(entry, 'atom:published', namespace),
                    'primary_category': get_text_from_element(entry.find('atom:primary_category', namespace), '{http://arxiv.org/schemas/atom}term', {}) if entry.find('atom:primary_category', namespace) is not None else None
                })

            print(f"Fetched {len(entries)} papers. Total for category: {len(papers)}")
            start += page_size
            
            # Be respectful to the API and wait between requests
            time.sleep(3)

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            break
            
    return papers[:max_results]

# --- Main script execution ---
categories = ['cs.AI', 'cs.CL', 'cs.LG']
all_paper_data = []

for cat in categories:
    papers_in_cat = fetch_papers_for_category(cat)
    all_paper_data.extend(papers_in_cat)
    print(f"Finished fetching for '{cat}'. Total papers so far: {len(all_paper_data)}")

print(f"\nTotal papers found across all categories: {len(all_paper_data)}")

df = pd.DataFrame(all_paper_data)
df['summary'] = df['summary'].str.replace('\n', ' ')
df.dropna(subset=['id', 'title', 'summary'], inplace=True) # Drop rows where essential info is missing

output_path = 'arxiv_papers.csv'
df.to_csv(output_path, index=False)

print(f"Data successfully saved to {output_path}")
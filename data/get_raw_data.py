import pandas as pd
import requests
import time
import os

url_base = "https://api.jikan.moe/v4/anime"
all_data = []
checkpoint_file = "remaining_pages.csv"

# Check if checkpoint_file already exists
if os.path.exists(checkpoint_file):
    pages_to_fetch = pd.read_csv(checkpoint_file)['page_number'].tolist()
    print(f"Resuming search: {len(pages_to_fetch)} pages remaining.")
else:
    response = requests.get(f"{url_base}", timeout=15)
    response.raise_for_status()
    last_page = response.json()["pagination"]["last_visible_page"]
    pages_to_fetch = list(range(1, last_page + 1))
    print(f"Starting search: {last_page} pages to fetch.")


for index, page in enumerate(list(pages_to_fetch)): # uses a copy of pages_to_fetch:
    if index % 25 == 0: print(f"Checkpoint: In page {page}.")
    try:
        response = requests.get(url_base, params={"page": page}, timeout=15)
        response.raise_for_status()
        
        page_data = response.json()
        all_data.extend(page_data["data"])
        
        pages_to_fetch.remove(page)
        
        time.sleep(1)
    
    except requests.exceptions.RequestException as e:
        print(f"Failed for page {page}: {e}")
        time.sleep(1)
        

if all_data:
    df = pd.json_normalize(all_data)
    if 'synopsis' in df.columns:
        df['synopsis'] = df['synopsis'].str.replace('\n', ' ', regex=False)
    
    file_exists = os.path.isfile("anime_raw_data.csv")
    df.to_csv("anime_raw_data.csv", mode='a', index=False, header=(not file_exists))
    

pd.DataFrame(pages_to_fetch, columns=['page_number']).to_csv(checkpoint_file, index=False)
  


        

import pandas as pd
from scrapy.crawler import CrawlerProcess
from extractURL_spider import ExtractUrls

# Read the Excel file
df = pd.read_excel('/Users/sribalac/Documents/Sustainathon Hackathon/Sustainathon Backend/scrapy/Scrapy_NGO.xlsx')

# Replace NaN values with an empty string
df = df.fillna('')

# Initialize a dictionary to store the combined NGO data
ngo_data = {}

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    urls = row['NGO WEBSITE'].split(',') if row['NGO WEBSITE'] else []
    state = row['STATE']
    sanctuary = row['SANCTUARY']
    state_sanctuary = f"{sanctuary} in {state}"
    ngos = row['NGO'].split(',') if row['NGO'] else []

    # Combine the data into a dictionary and add to the ngo_data dictionary
    for url, ngo in zip(urls, ngos):
        if url not in ngo_data:
            ngo_data[url] = {}
        if ngo not in ngo_data[url]:
            ngo_data[url][ngo] = {
                'state_sanctuaries': []
            }
        ngo_data[url][ngo]['state_sanctuaries'].append(state_sanctuary)

# Initialize the Scrapy process
# process = CrawlerProcess(settings={
#     "FEEDS": {
#         "output.json": {"format": "json"},
#     },
# })
process = CrawlerProcess()

# Iterate over the ngo_data dictionary and call the ExtractUrls spider
for url, ngos in ngo_data.items():
    for ngo, data in ngos.items():
        state_sanctuaries = ', '.join(data['state_sanctuaries'])
        process.crawl(ExtractUrls, urls=[url], exclude_strings=[], state=[], sanctuary=[state_sanctuaries], ngos=[ngo])

# Start the Scrapy process
process.start()
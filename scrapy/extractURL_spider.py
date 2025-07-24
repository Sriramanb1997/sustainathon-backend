import os
import sys
import scrapy
from scrapy.crawler import CrawlerProcess
import logging
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse, urljoin
import argparse
import json

# Add parent directory to Python path to import db module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db import store_contexts_objects, get_context_collection_count


class ExtractUrls(scrapy.Spider):
    name = "extract"

    def __init__(self, urls, exclude_strings, state, sanctuary, ngos,  output_dir="output", *args, **kwargs):
        super(ExtractUrls, self).__init__(*args, **kwargs)
        self.urls = urls
        self.exclude_strings = exclude_strings
        self.state = state
        self.sanctuary = sanctuary
        self.ngos = ngos
        self.results = {}
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)  # Create the output folder if it doesn't exist


    def start_requests(self):
        #urls = ['https://www.bnhs.org/listing/conservation-research', ]
        for url in self.urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        try:
            if 'text' not in response.headers.get('Content-Type', b'').decode('utf-8'):
                logging.warning(f"Skipping non-text response from {response.url}")
                return

            html_content = response.css('body').get()
            if not html_content:
                logging.error(f"Failed to extract body content from {response.url}")
                return


            soup = BeautifulSoup(html_content, "html.parser")
            body_content = soup.body
            if not body_content:
                logging.error(f"Failed to extract body content from {response.url}")
                return

            all_text = body_content.get_text(separator=' ', strip=True)
            all_text = re.sub(r'\n', ' ', all_text)
            all_text = re.sub(r'[^\w\s]', '', all_text)
            all_text = re.sub(r'\s+', ' ', all_text).strip()

            # Debug: Log content length before processing
            print(f"Raw content length for {response.url}: {len(all_text)} characters")
            print(f"First 200 chars: {all_text[:200]}...")

            text_item = {
                'url': response.url,
                'sanctuary': self.sanctuary,
                'ngos': self.ngos,
                'content': all_text
            }

            domain = urlparse(response.url).netloc
            if domain not in self.results:
                self.results[domain] = []
            self.results[domain].append(text_item)
            
            # Debug: Check count before storing
            count_before = get_context_collection_count()
            print(f"Count before storing {response.url}: {count_before}")
            
            # Store the context
            success = store_contexts_objects(text_item)
            print(f"Storage success for {response.url}: {success}")
            
            # Debug: Check count after storing
            count_after = get_context_collection_count()
            print(f"Count after storing {response.url}: {count_after}")
            
            yield text_item

            base_url = response.url
            links = response.css('a::attr(href)').extract()
            for link in links:
                full_link = urljoin(base_url, link)
                if full_link.startswith(base_url) and not any(exclude in full_link for exclude in self.exclude_strings):
                    logging.info(f"Requesting URL: {full_link}")
                    yield scrapy.Request(url=full_link, callback=self.parse)

        except Exception as e:
            logging.error(f"Error processing {response.url}: {e}")

    # def closed(self, reason):
    #     for domain, items in self.results.items():
    #         filename = os.path.join(self.output_dir, f"{domain}.json")
    #         with open(filename, 'w') as f:
    #             json.dump(items, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scrape URLs.')
    parser.add_argument('urls', type=str, help='Comma-separated list of URLs to scrape')
    parser.add_argument('--exclude', type=str, help='Comma-separated list of strings to exclude from links', default='')
    parser.add_argument('--state', type=str, help='Comma-separated list of State name',default='')
    parser.add_argument('--sanctuary', type=str, help='Comma-separated list of Sanctuary name',default='')
    parser.add_argument('--ngos', type=str, help='Comma-separated list of NGOs',default='')
    args = parser.parse_args()

    url_list = args.urls.split(',')
    exclude_list = args.exclude.split(',') if args.exclude else []
    state_list = args.state.split(',') if args.state else []
    sanctuary_list = args.sanctuary.split(',') if args.sanctuary else []
    ngo_list = args.ngos.split(',') if args.ngos else []

    process = CrawlerProcess(settings={
        "FEEDS": {
            "output.json": {"format": "json"},
        },
        "TELNETCONSOLE_ENABLED": False,  # Disable telnet console to avoid shutdown errors
    })
    process.crawl(ExtractUrls, urls=url_list, exclude_strings=exclude_list, state=state_list, sanctuary=sanctuary_list,
                  ngos=ngo_list)
    process.start()
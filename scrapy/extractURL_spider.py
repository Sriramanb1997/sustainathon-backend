import scrapy
from scrapy.crawler import CrawlerProcess
import logging
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse, urljoin
import argparse
import json

from db import store_contexts_objects


class ExtractUrls(scrapy.Spider):
    name = "extract"

    def __init__(self, urls, exclude_strings, state, sanctuary, ngos, *args, **kwargs):
        super(ExtractUrls, self).__init__(*args, **kwargs)
        self.urls = urls
        self.exclude_strings = exclude_strings
        self.state = state
        self.sanctuary = sanctuary
        self.ngos = ngos
        self.results = {}

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
            store_contexts_objects(text_item)
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
    #         filename = f"{domain}.json"
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
    })
    process.crawl(ExtractUrls, urls=url_list, exclude_strings=exclude_list, state=state_list, sanctuary=sanctuary_list,
                  ngos=ngo_list)
    process.start()
'''
This function scrapes ticker descriptions from google finance.
Note: Yahoo finance loads the data through JS making it more difficult to scrape, so Google Data here should be fine.
Author: Adam Lieberman
'''
from bs4 import BeautifulSoup
import requests
def scrape_google_description(tickers):
    descriptions = []
    for ticker in tickers:
        myurl = 'https://www.google.com/finance?q='+ticker+'&ei=ZqMsWMCFE8SOmAHPrJS4DQ'
        #myurl = "https://www.google.com/finance?q=NASDAQ%3A"+ticker+"&ei=u28rWLmuIsTumAHWu6WABQ"
        html = requests.get(myurl).content
        soup = BeautifulSoup(html,"html.parser")
        description = soup.find('div', attrs={'class':'companySummary'})
        description =  list(description)[0]
        descriptions.append(description)
    return descriptions

def scrape_google_sector(tickers):
    sectors = []
    for ticker in tickers:
        myurl = 'https://www.google.com/finance?q='+ticker+'&ei=ZqMsWMCFE8SOmAHPrJS4DQ'
        html = requests.get(myurl).content
        soup = BeautifulSoup(html,"html.parser")
        description = soup.find('a', attrs={'id':'sector'})
        description =  list(description)[0]
        sectors.append(description)
    return sectors
if __name__ == '__main__':
    tickers = ['AAPL','BAC']
    #descriptions = scrape_google_description(tickers)
    #print descriptions
    #print descriptions[1]
    sectors = scrape_google_sector(tickers)
    print sectors
    print sectors[0]
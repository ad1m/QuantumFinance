from selenium import webdriver
from PIL import Image
import StringIO
import time
'''
Author: Adam Lieberman
'''

#Scrapes Analyst Estimate Charts from yahoo finance by screenshotting the charts and then cropping them.
def scrape_estimates(stocks):

    for i in stocks:
        driver = webdriver.PhantomJS() # the normal SE phantomjs binding
        driver.set_window_size(2024, 2000)
        #driver.get('https://google.com/') # whatever reachable url
        #driver.execute_script("document.write('{}');".format(htmlString))  # changing the DOM
        driver.get('https://finance.yahoo.com/quote/'+str(i)+'/analysts?p='+str(i))
        time.sleep(5)
        driver.save_screenshot('screen.png')   #screen.png is a big red rectangle :)
        screen =  driver.get_screenshot_as_png()
        #box = (1000,900,1800,1400) Left, Upper, Right, Lower

        #Analyst Price Targets: box = (1300,970,1680,1070)
        analyst_price_targets = (1300,970,1680,1075)
        recommendation_rating = (1300,800,1630,905)
        #box = (1300,530,1630,772)
        im = Image.open(StringIO.StringIO(screen))
        #region = im.crop(box)
        region1 = im.crop(analyst_price_targets)
        region2 = im.crop(recommendation_rating)
        #region.save('test.jpg','JPEG',optimize=True,quality=150)
        region1.save('t.jpg','JPEG',optimize=True,quality=150)
        #driver.quit()

        #print "png file created"

if __name__ =='__main__':
    scrape_estimates(['AAPL'])
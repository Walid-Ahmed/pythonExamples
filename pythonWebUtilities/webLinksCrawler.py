

from urllib.request import urlopen
from bs4 import BeautifulSoup

import ssl

def hmllLinkCrawler(url,count,position):
        count=count-1
        print ("Retrieving: "+url)
        if(count<0):
    	     return
        html = urlopen(url, context=ctx).read()
        soup = BeautifulSoup(html, "html.parser")
        tags = soup('a')
        i=0
        for tag in tags:
        	i=i+1
        	if (i==position):
        		url=tag.get('href', None)
        		return hmllLinkCrawler(url,count,position)  


# Ignore SSL certificate errors
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url=input("Enter URL: ")

count=int(input("Enter count: "))
position=int(input("Enter position: "))


hmllLinkCrawler(url,count,position)
 
    	

import urllib.request, urllib.parse, urllib.error
import xml.etree.ElementTree as ET


url = input('Enter url for xml file: ')
print('Retrieving', url)
uh = urllib.request.urlopen(url)
data = uh.read()
print('Retrieved', len(data), 'characters')
print(data.decode())
tree = ET.fromstring(data)
counts = tree.findall('.//count')
print ("Count:"+str(len(counts)))
sum=0
for count in counts:
	sum=sum+int(count.text)
print ("Sum:"+str(sum))


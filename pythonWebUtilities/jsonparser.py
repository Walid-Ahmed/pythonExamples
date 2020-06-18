import json
import urllib.request, urllib.parse, urllib.error

data = '''
[
  { "id" : "001",
    "x" : "2",
    "name" : "Chuck"
  } ,
  { "id" : "009",
    "x" : "7",
    "name" : "Chuck"
  }
]
'''

url=input("Please enter location: ")  # http://py4e-data.dr-chuck.net/comments_42.json
print("Retrieving "+ url)
uh = urllib.request.urlopen(url)
data = uh.read().decode()
info = json.loads(data)
print('Retrieved:'+ str ( len(info) ) )

#print(info)
x=info["comments"]
print("Count: "+str(len(x)))
sum=0


for item in x:
    sum=sum+ item['count']
print("Sum: "+str(sum))
   
   

'''
Implement an algorithm to determine if a string has all unique characters. What if you can not use additional data structures?
'''
from collections import deque

def isStringUniqueCharcters (testString):
    # Assuming character set is ASCII (128 characters)
    if len(testString) > 128:
        return False
    for char in testString:
        #print(char)
        if (testString.count(char) >1):
            return False

    return True



'''
Write a method to decide if two strings are anagrams or not
'''
def  isAnagrams(stringA,stringB):
    if len(stringA) != len(stringB):
        return False
    for i in range (len(stringB)): 
        #print(i)   #0
        j=len(stringB)-i-1   #n
        print(i,j)
        if stringA[i]!=stringB[j]:
            return False
    return True

'''
Design an algorithm and write code to remove the duplicate characters in a string
'''
def removeStringDuplicates(testString):
    if (isStringUniqueCharcters (testString)):
            return testString
    newString=[]
    for char in testString:
            if char in newString:
                continue
            else:
                newString.append(char)
    
    return  ''.join(newString)

'''
Write a method to replace all spaces in a string with ‘%20’.

'''
def replaceSpace(testString):
    newString=[]
    for char in  testString:
        if (char==" "):
            newString.append("%20")
        else:
            newString.append(char)  
    return    ''.join(newString)     
            

def shift(l, n):
     return l[n:] + l[:n]

def checkRotation(StringA,StringB):
    newString=[]
    fullString=[]
    for  char in StringA:
        newString.append(char)   
    items = deque(newString)
    for i in range (len(StringA)):
         items.rotate(1)
         lst1=list(items)
         fullString=fullString+lst1
    fullString=''.join(fullString)
    return  (StringB in (fullString))


#flag=isStringUniqueCharcters("Helo")
#print(flag)
#stringA="abcd"
#stringB="dcba"
#flag=isAnagrams(stringA,stringB)
#print(flag)
testString="aaabbbcccaaa"
newString=removeStringDuplicates(testString)
#print(newString)
#testString="aaa bbb ccc"
#print(replaceSpace(testString))
stringA="waterbottle"
stringB="erbottlewat"
flag=checkRotation(stringA,stringB)
print(flag)

input("press any key to continue")

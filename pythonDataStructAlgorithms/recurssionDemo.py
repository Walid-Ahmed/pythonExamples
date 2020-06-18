
#Converting an Integer to a String in Any Base
def to_str(n, base):
    #print(n)
    convert_string = "0123456789ABCDEF"
    if n < base:

            return convert_string[n]
    else:
            print(n,base)
            print(n % base)
            
            return to_str(n // base, base) + convert_string[n % base]

#print(to_str(1453, 16))


#Write a function that takes a string as a parameter and returns a new string that is the reverse of the old string.
def reverseString(str):
    if len(str)==1:
        return str
    else:
        char=str[len(str)-1]
        newStr=str[0:(len(str)-1)]
        return char+reverseString(newStr)

    


#print(reverseString("ABCDE"))    

'''
Write a function that takes a string as a parameter and returns True if the string is a palindrome,
False otherwise. Remember that a string is a palindrome if it is spelled the same both forward
and backward. for example: radar is a palindrome. for bonus points palindromes can also be
phrases, but you need to remove the spaces and punctuation before checking. for example:
madam i’m adam is a palindrome.
'''

def checkPalindrome(testString):
    a=testString[0]
    b=testString[len(testString)-1]
    print(a,b)
    if (a!=b):
        print("2 charcers not equal")
        return False
    else:  #(a=b)
        testString=testString[1:(len(testString)-1)]
        print(testString)
        if len(testString)==1:
            return True
        return (True and checkPalindrome(testString))
        

#print(checkPalindrome("radar"))

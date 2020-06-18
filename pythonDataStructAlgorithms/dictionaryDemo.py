#histogram
word = 'brontosaurus'
d = dict()
for c in word:
  d[c] = d.get(c,0) + 1
print(d)


counts = { 'chuck' : 1 , 'annie' : 42, 'jan': 100}
for key in counts:
  print(key, counts[key])
  
  

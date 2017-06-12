import os
 
data = '101_ObjectCategories'
path = os.listdir(data) 
path.sort()
vp = 0.105 
file = open('train.txt','w')
fv = open('val.txt','w')
i = 0
 
for line in path:
    subdir = data +'/'+line
    childpath = os.listdir(subdir)
    mid = int(vp*len(childpath))
    for child in childpath[:mid]:
        subpath = data+'/'+line+'/'+child;
        d = ' %s' %(i)
        t = subpath + d
        fv.write(t +'\n')
    for child in childpath[mid:]:
        subpath = data+'/'+line+'/'+child;
        d = ' %s' %(i)
        t = subpath + d
        file.write(t +'\n')
    i=i+1
 
file.close()
fv.close()
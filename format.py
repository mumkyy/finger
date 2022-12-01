import csv
import os



f = open('hlabels.csv')
reader = list(csv.reader(f))
labels = []
row = 1
for file in os.listdir('d'):
    points = []
    line = reader[row]
    while file == line[0]:
        x = line[5].split(':')[2]
        y = line[5].split(':')[3]
        x = int(''.join(z for z in x if z.isdigit()))
        y = int(''.join(z for z in y if z.isdigit()))
        points.append(x)
        points.append(y)
        row +=1
        if row < 2110:
            line = reader[row]
        else:
            break
    if points:
        points.insert(0,file)
        labels.append(points)
        
g = open('rlabels.csv','r+',newline='')
writer = csv.writer(g)
writer.writerows(labels)
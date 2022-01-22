import csv

csvName = "response.csv"

def writeToCSV(array):
    with open(csvName, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        writer.writerow(array)

def readFromCSV(array):
    with open(csvName, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', lineterminator='\n')
        next(reader)
        for row in reader:
            array.append(row)

array = ["tsne",5,3,"testing,"]
writeToCSV(array)

outputArray = []
readFromCSV(outputArray)

for idx in range(len(outputArray)):
    print(outputArray[idx])

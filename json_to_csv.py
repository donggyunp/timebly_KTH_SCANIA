import json
import csv
import sys
 
def main(json_file_path, csv_file_path):    
    with open(json_file_path) as json_file:
        data_json = json.load(json_file)
    
    data = data_json['annotation']['actionAnnotationList']
    data_csv = open(csv_file_path, 'w')
    
    csv_writer = csv.writer(data_csv)
    
    # Counter variable used for writing
    # headers to the CSV file
    count = 0
    
    for ele in data:
        if count == 0:
    
            # Writing headers of CSV file
            header = ele.keys()
            csv_writer.writerow(header)
            count += 1
    
        # Writing data of CSV file
        csv_writer.writerow(ele.values())
    
    data_csv.close()

if __name__ == "__main__":
    json_file_path = sys.argv[1]
    csv_file_path = sys.argv[2]
    main(json_file_path, csv_file_path)
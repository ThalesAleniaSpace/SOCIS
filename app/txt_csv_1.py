
import csv
import os 
import codecs
# from pyth.plugins.rtf15.reader import Rtf15Reader
# from pyth.plugins.plaintext.writer import PlaintextWriter
# create the labels of the csv [ s.no, id, date, x1, y1,...............x10000, y10000 ]
labels = [['s.no', 'id']]

for i in range(10000):

    labels[0].append('x'+ str(i+1))
    labels[0].append('y'+ str(i+1))

#list of labels

# print(labels)
# labels[0] = labels[0] + ['a', 'a', 'a', 'a', 'a','a','a','a','a','a','a','a','a','a','a','a','a','a','a','a','a','a','a','a',]
csvData = labels
# print(len(labels[0]))
#creating a .csv and adding labels

with open('input/points.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csvData)

#adding data into labels

ROOT_DIR = os.path.abspath("./")
REQ_DIR = os.path.join(ROOT_DIR,'Dataset')
REQ_DIR =  os.path.join(REQ_DIR,'ON_OFF_Consumption')

# print(REQ_DIR,"REQ_Directory")
 
# print(os.walk(REQ_DIR))

# find all directories
all_tests = []

for (dirpath, dirnames, filenames) in os.walk(REQ_DIR):
    all_tests.extend(dirnames)
    break


# path of all directories

all_test_dir_path = []

for i in all_tests:

    all_test_dir_path.append(os.path.join(REQ_DIR,str(i)))


# files in each dir

all_files = []
# print(all_tests)
for i in all_test_dir_path:
    temp = []
    # print(i)
    for (dirpath, dirnames, filenames) in os.walk(i):
        
        temp.extend(filenames)
        break
    all_files.append(temp)
        

# path of all files
all_files_path = []

for i in range(len(all_test_dir_path)):
    temp = []
    for j in range(len(all_files[i])):
        temp.append(os.path.join(all_test_dir_path[i],all_files[i][j]))

    all_files_path.append(temp)

# print(all_files_path)

#////////////////////////////////extracting from the files


def getpoints(filepath, dir1):
    

    filepath = filepath
    fo = open(filepath, "r")

    filename = os.path.basename(fo.name)
    test = "HEGSE"
    if filename.find(test) == -1:
        

        # print (PlaintextWriter.write(doc).getvalue()) - failed
        # file = open(filepath,'r',errors="ignore") -failed
        # content = file.readlines() - failed

        lines = [line.rstrip('\n') for line in open(filepath,'r',errors="ignore")]

        
    # print(type(lines))

    filename_str = "File Name:"
    filedate_str = "File Date:"
    filetrace_str= "[GRAPH_01\TRACES]"
    filetrace_end = "------------------------------------------------------------------------------"


    filename_index = 4
    filedate_index = 5
    filetrace_index = 103
    for index in range(len(lines)):

        temp_str = lines[index]
        if temp_str.find(filename_str) != -1:
            
            filename_index = index
        if temp_str.find(filedate_str) != -1:
            filedate_index = index
        if temp_str.find(filetrace_str) != -1:
            filetrace_index = index
            
        # if temp_str.rfind(filetrace_end) != -1:
        #     print("p")
        #     filetraceend_index = index
        #     break

        
    filetraceend_index =  len(lines)-1

   
    # find file name 
    temp_str = lines[filename_index].strip(filename_str)
    final_filepath = temp_str
    # print(temp_str)

    k1 = temp_str.rfind("\\")
    
    k2 = temp_str.find(".sfg")

    final_filename = temp_str[k1+1:k2]
    # print(final_filename)

    # find file date
    temp_date = lines[filedate_index].strip(filedate_str)
    final_filedate = temp_date
    # print(final_filedate)

    # find file points

    start_index = filetrace_index  + 2

    end_index = filetraceend_index -2

    trace_points = []
    # print(start_index,"s")
    # print(end_index,"s")
    for ind in range(start_index,end_index,1):
        temp_trace = lines[ind]
        com_i = temp_trace.find(",")
        x = temp_trace[0:com_i]
        y = temp_trace[com_i+1:]
        # print(x,"---",y)
        trace_points.append(float(x))
        trace_points.append(float(y))
    # print(trace_points)

    return_list = []
    return_list.append(sno)
    return_list.append(final_filename)
    # return_list.append(dir1)
    # return_list.append(final_filepath)
    # return_list.append(final_filedate)
    return_list = return_list + trace_points
    # print(return_list[0:20])
    
    fo.close()

    # with open('your_file.txt', 'w') as f:
    #     for item in range(len(lines)):
    #         f.write("%s" % item)
    #         f.write("%s\n" % lines[item])
    return return_list
    
# call the function

csv_list = []
sno = 0
for i in range(len(all_files_path)):
    for j in range(len(all_files_path[i])):
        
 
        filepath = all_files_path[i][j]
        filepath = filepath
        fo = open(filepath, "r")

        filename = os.path.basename(fo.name)
        test = "HEGSE"
        if filename.find(test) == -1:
            sno = sno+1
            return_list = getpoints(filepath,1)
            csv_list.append(return_list)

# with open('points.csv', 'a') as csvFile:
#     writer = csv.writer(csvFile)
#     writer.writerows(csv_list)

    
        
    
        
        

# with open('your_file_1.txt', 'w') as f:
#         for item in range(len(csv_list[0])):
#             # f.write("%s" % item)
#             f.write("%s\n" % csv_list[item])
    

    
    







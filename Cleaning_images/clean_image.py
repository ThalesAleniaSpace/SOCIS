import csv
import os 
import codecs



#adding data into labels

ROOT_DIR = os.path.abspath("./")
REQ_DIR = os.path.join(ROOT_DIR,'Dataset')
REQ_DIR =  os.path.join(REQ_DIR,'ON_OFF_Consumption')

print(REQ_DIR,"REQ_Directory")
 
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



def cleanimage(filepath,sno, dir1):




    

    filepath = filepath
    fo = open(filepath, "r+")

    filename = os.path.basename(fo.name)
    test = "HEGSE"
    if filename.find(test) == -1:
        

        # print (PlaintextWriter.write(doc).getvalue()) - failed
        # file = open(filepath,'r',errors="ignore") -failed
        # content = file.readlines() - failed

        lines = [line.rstrip('\n') for line in open(filepath,'r',errors="ignore")]

        
    # print((lines))

    filename_str = "File Name:"
    filedate_str = "File Date:"
    filetrace_str= "[GRAPH_01\TRACES]"
    filetrace_label = " [GRAPH_01\LABELS]"
    file_mouse = "[GRAPH_01\MOUSE]"
    file_X ="Axis_X"
    file_Y="Axis_Y"
    # filetrace_end = "------------------------------------------------------------------------------"



    filelabel_index = 54
    filemouse_index = 88
    filex_index = 43
    filey_index= 44
    for index in range(len(lines)):

        temp_str = lines[index]

        if temp_str.find(file_X) != -1:
            
            filex_index = index


        if temp_str.find(file_Y) != -1:
            
            filey_index = index
        if temp_str.find(file_mouse) != -1:
            
            filemouse_index = index
        if temp_str.find(filetrace_label) != -1:
            
            filelabel_index = index
        if temp_str.find(filename_str) != -1:
            
            filename_index = index
        if temp_str.find(filedate_str) != -1:
            filedate_index = index
        if temp_str.find(filetrace_str) != -1:
            filetrace_index = index
           
    lines[filelabel_index:filemouse_index]=[]
    lines[filex_index]= 'Axis_X      = INDEX, 0, 0, 0, 0, 0, 0'
    lines[filey_index]= 'Axis_Y      = STANDARD, 0, 0, 0, 0, 0, 0'
    fo.close()
    with open(filepath, 'w') as f:
        for item in lines:
            f.write("%s\n" % item)



# # call the function


sno = 0
for i in range(len(all_files_path)):
    for j in range(len(all_files_path[i])):
        
 
        filepath = all_files_path[i][j]
        filepath = filepath
        fo = open(filepath, "r")
        
       

        filename = os.path.basename(fo.name)
        test = "HEGSE"
      
        if filename.find(test) == -1:
            print(filename)
            sno = sno+1
            cleanimage(filepath,sno,all_tests[i])


    
    







import csv
import os 
import codecs
# from pyth.plugins.rtf15.reader import Rtf15Reader
# from pyth.plugins.plaintext.writer import PlaintextWriter
# create the labels of the csv [ s.no, id, date, x1, y1,...............x10000, y10000 ]
labels = [['s.no', 'id', 'dir', '_file_', 'power_state_value','current_rise/fall_time_value (mS)','current_stabilised_value (mA)','current_max/min_value (mA)', 'power_state_spec','current_rise/fall_time_spec (mS)','current_stabilised_spec (mA)','current_max/min_spec (mA)', 'power_state_N/NC' ,'current_rise/fall_time_C/NC','current_stabilised_C/NC','current_max/min_C/NC']]


#list of labels

# print(labels)
csvData = labels

#creating a .csv and adding labels

with open('values.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csvData)

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
print(all_tests)
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



def  getvalues(sno,final_filetype,lines,powerstate_index,dir1,final_index):
    sno = sno+1

    if(dir1 =="000"):
        print("00")
        final_filename = " "
    else:

        filename_index = powerstate_index-1
        temp_str = lines[filename_index].strip()
        k1 = temp_str.rfind("=")
        k2 = temp_str.find(".sfg")
        final_filename =  temp_str[k1+1:k2]
    
    # powervalues,spec,c/cn
    for i in range(3):
        ind = powerstate_index+i+1

        if (ind+i)>=final_index:

            temp_str=''
        else:
            temp_str = lines[ind].strip()
        
        k1 = temp_str.rfind("=")
        if (i==0):

            if (ind+i)>=final_index:
                final_power_val=''
            else:
                final_power_val = temp_str[k1+1:].strip()


            
        if (i==1):

            if (ind+i)>=final_index:
                final_power_spec =''
            else:
                final_power_spec = temp_str[k1+1:].strip()
            
            
        if (i==2):


            if (ind+i)>=final_index:
                final_power_nc =''
            else:
                final_power_nc = temp_str[k1+1:].strip()


           
    # print(final_power_val,"-",final_power_spec,"-",final_power_nc)
    # currenttime val,spec,c/nc
    currenttime_index = powerstate_index + 4
    temp_x = lines[currenttime_index].strip()
    x = temp_x.find("=")
    y = temp_x[x+1:]
    if(y!="V PRIM CURRENT RISE TIME" and  y!="V PRIM CURRENT FALL TIME"):
        print("l")
        print(final_filename)
        final_currenttime_val=''
        final_currenttime_spec=''
        final_currenttime_nc=''
        currenttime_index=powerstate_index
    else :
        for i in range(3):
            ind = currenttime_index +i+1
            if (ind+i)>=final_index:
                temp_str=''
            else:
                temp_str = lines[ind].strip()

            k1 = temp_str.rfind("=")
            if (i==0):
                if (ind+i)>=final_index:
                    final_currenttime_val=''
                else:
                    final_currenttime_val = temp_str[k1+1:].strip()
                
            if (i==1):
                if (ind+i)>=final_index:
                    final_currenttime_spec=''
                    
                else:
                    k2 = temp_str.rfind("mS")
                    final_currenttime_spec = temp_str[k1+2:k2-1].strip()
            if (i==2):
                if (ind+i)>=final_index:
                    final_currenttime_nc=''
                else:
                    # k2 = temp_str.rfind("mS")
                    final_currenttime_nc = temp_str[k1+1:].strip()
    # print(final_currenttime_val,"-",final_currenttime_spec,"-",final_currenttime_nc)
    # curretstable val,spec,c/nc
    currentstable_index = currenttime_index + 4

    temp_x = lines[currentstable_index].strip()
    x = temp_x.find("=")
    y = temp_x[x+1:]
    if(y!="V PRIM CURRENT STABILIZED"):
        final_currentstable_val=''
        final_currentstable_spec=''
        final_currentstable_nc=''
        currentstable_index=currenttime_index
    else:

        for i in range(3):
            ind = currentstable_index +i+1

            if (ind+i)>=final_index:
                    temp_str=''
            else:
                temp_str = lines[ind].strip()
        
            k1 = temp_str.rfind("=")
            if (i==0):

                if (ind+i)>=final_index:
                        final_currentstable_val=''
                else:
                    final_currentstable_val= temp_str[k1+1:].strip()


                
            if (i==1):

                if (ind+i)>=final_index:
                        final_currentstable_spec=''
                else:
                    k2 = temp_str.rfind("mA")
                    final_currentstable_spec = temp_str[k1+2:k2-1].strip()

                
            if (i==2):

                if (ind+i)>=final_index:
                    final_currentstable_nc=''
                else:
                    final_currentstable_nc = temp_str[k1+1:].strip()

            
    # print(final_currentstable_val,"-",final_currentstable_spec,"-",final_currentstable_nc)
    # curretminmax val,spec,c/nc
    currentminmax_index = currentstable_index + 4


    temp_x = lines[currentminmax_index].strip()
    x = temp_x.find("=")
    y = temp_x[x+1:]
    if(y!="V PRIM CURRENT MAX" and y!= "V PRIM CURRENT MIN"):
        final_currentminmax_val=''
        final_currentminmax_spec=''
        final_currentminmax_nc=''
        currentminmax_index=currentstable_index
    else:

        for i in range(3):
            ind = currentminmax_index +i+1

            if (ind+i)>=final_index:
                    temp_str =''
            else:
                temp_str  = lines[ind].strip()

            
            k1 = temp_str.rfind("=")
            if (i==0):

                if (ind+i)>=final_index:
                    final_currentminmax_val =''
                else:
                    final_currentminmax_val = temp_str[k1+1:].strip()
                
            if (i==1):

                if (ind+i)>=final_index:
                    final_currentminmax_spec =''
                else:
                    k2 = temp_str.rfind("mA")
                    final_currentminmax_spec = temp_str[k1+2:k2-1].strip()
                
            if (i==2):

                if (ind+i)>=final_index:
                    # print(ind+i)
                    # print(final_index)
                    final_currentminmax_nc =''
                else:
                    # print(temp_str)
                    final_currentminmax_nc = temp_str[k1+1:].strip()
                    # print(final_currentminmax_nc)

                
    # print(final_currentminmax_val,"-",final_currentminmax_spec,"-",final_currentminmax_nc)

    return_list = []
    return_list.append(sno)
    return_list.append(final_filename)
    return_list.append(dir1)
    return_list.append(final_filetype)
    
    return_list.append(final_power_val)
    return_list.append(final_currenttime_val)
    return_list.append(final_currentstable_val)
    return_list.append(final_currentminmax_val)
    return_list.append(final_power_spec)
    return_list.append(final_currenttime_spec)
    return_list.append(final_currentstable_spec)
    return_list.append(final_currentminmax_spec)
    return_list.append(final_power_nc)
    return_list.append(final_currenttime_nc)
    return_list.append(final_currentstable_nc)
    return_list.append(final_currentminmax_nc)

    # print(return_list,"return_list")
    return return_list,sno



def getpoints(filepath,sno,dir1):
    
    

    filepath = filepath
    fo = open(filepath, "r")

    filename = os.path.basename(fo.name)
    test = "HEGSE"
    if filename.find(test) != -1:

        lines = [line.rstrip('\n') for line in open(filepath,'r',errors="ignore")]

        
    

    filetype_str = "_File_     ="
    filetype_index = 9
    for index in range(len(lines)):

        temp_str = lines[index]
        if temp_str.find(filetype_str) != -1:
            
            filetype_index = index
            break
       
        
    filetraceend_index =  len(lines)-1
    # if(lines[filetraceend_index-8].strip()==)
    final_index = filetraceend_index - 6
    
    temp_str = lines[index].strip()
    k1 = temp_str.rfind("=")
    
    # k2 = temp_str.find(".sfg")

    final_filetype = temp_str[k1+1:]
    # print(final_filetype)
    print(dir1)

    power_state = "HILINK POWER STATE"
    return_list1 = []
    for index in range(len(lines)):
        temp_str = lines[index]
        if temp_str.find(power_state) != -1:
            powerstate_index= index
            
            return_list ,sno = getvalues(sno,final_filetype,lines,powerstate_index,dir1,final_index)
            return_list1.append(return_list)
            
    fo.close()
    return return_list1,sno

    # with open('your_file.txt', 'w') as f:
    #     for item in range(len(lines)):
    #         f.write("%s" % item)
    #         f.write("%s\n" % lines[item])
    # return return_list
###################################################################################    
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
        if filename.find(test) != -1:
            
            return_list1,sno = getpoints(filepath,sno,all_tests[i])
            # print(return_list1)
            # csv_list.append(return_list)
            csv_list = csv_list + return_list1
            # print(csv_list)
        
    

with open('values.csv', 'a') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csv_list)

    
        
    
        
        

# # with open('your_file_1.txt', 'w') as f:
# #         for item in range(len(csv_list[0])):
# #             # f.write("%s" % item)
# #             f.write("%s\n" % csv_list[item])
    

    
    







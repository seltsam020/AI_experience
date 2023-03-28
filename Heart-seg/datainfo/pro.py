import shutil
import os
source="D:/heartseg/Heart Data/"
dst_file= "/"
path1=os.listdir(source)
print(path1)
for i in path1:
    path2=source+i+"/png/Label"
    print(path2)
    list=os.listdir(path2)
    print(list)
    for j in list:
        path3=path2+"/"+j
        names=os.listdir(path3)
        #print(path3)
        #print(names)
        for k in names:
            #pass
            shutil.copyfile(path3+"/"+k,"D:\heartseg\seg2/traindata/label/"+i+j+"No."+k[:-4][5:]+".png")
            #print(path3+"/"+k)
            print("D:\heartseg\seg2/traindata/label/"+i+j+"No."+k[:-4][5:])
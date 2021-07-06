from datasetConfig import dsconf
import subprocess

def makeohv(train,train_ver,test,test_ver,val=None,val_ver=None):
    ohvpath = dsconf["path"]+dsconf["ohvdir"]
    outname = getOhvDirName(train,train_ver,test,test_ver,val,val_ver)
    srcpath = dsconf["path"]+dsconf["srcdir"]


    call = ['java','-jar','ohv.jar']
    call += ['/'.join([srcpath,dsconf[train][train_ver]]),
             '/'.join([srcpath,dsconf[test][test_ver]]),
             '/'.join([srcpath,dsconf[val][val_ver]]),
             '/'.join([ohvpath,outname])]


    print("start : make ohv file")
    subprocess.call(call)
    print("end : make ohv file")
    return outname

def getOhvDirName(train,train_ver,test,test_ver,val=None,val_ver=None):
    outname = train[:2]+train_ver+test[:2]+test_ver
    if val is not None:
        outname += val[:2]+val_ver
    return outname


if __name__ == "__main__":
    makeohv("Xerces",dsconf["Xerces"]["versions"][1],"Xerces",dsconf["Xerces"]["versions"][2],"Synapse",dsconf["Synapse"]["versions"][2])

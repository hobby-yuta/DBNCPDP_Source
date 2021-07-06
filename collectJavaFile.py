from database import Database
import numpy as np
import csv
import os
import re
import config
from datasetConfig import dsconf

def collectJavaFile(projectfile):
    javafilelist = []
    for entry in os.scandir(projectfile):
        if entry.is_file():
            if entry.name.find('.java') != -1:
                javafilelist.append(entry.path.replace("\\","/"))

        else:
            javafilelist += collectJavaFile(projectfile+"/"+entry.name)

    return javafilelist

def extractPath(paths,root="org"):
    _paths = []
    for path in paths:
        if path.find(root) != -1:
            _paths.append(re.sub('.*?org','org',path,count=1))
    return _paths


"""
makelabeldata 引数の説明
    projectdir      バグを検出するプロジェクトのディレクトリのパス
    metricsfile     ダウンロードしたデータセットのcsvファイルのパス
    out             出力先のパス
"""
def makelabeldata(projectdir,metricsfile,out):
    jfiles = collectJavaFile(projectdir)
    with open(metricsfile, mode="r") as f:
        list = [row for row in csv.reader(f)]
        list = list[1:]
        #ファイル名は3列目
        names = [l[2] for l in list]
        names = [name.replace(".","/") + ".java" for name in names]
        _files = []

        for name in names:
            for file in jfiles:
                if file.find(name) != -1:
                    _files.append(name)

                    break
        jfiles = _files
        labels = np.full(len(jfiles),-1)
        #バグの数は最後の列
        bugs = [l[-1] for l in list]
        print(len(list),"metrics exists.")

    for i,file in enumerate(jfiles):
        for name,bug in zip(names,bugs):
            if file.find(name) != -1:
                labels[i] = 1 if int(bug) > 0 else 0


    count = 0
    with open(out, mode="w") as f:
        for file,label in zip(jfiles,labels):
            if label != -1:
                f.write(file+","+str(label)+"\n")
                count+=1
    print(count,"file collected.")

"""
def makeJavaFileDataset(projectfile,wfile,version):
    javafilelist = collectJavaFile(projectfile)
    javafilelist = [f.replace(projectfile+"/","") for f in javafilelist]
    label = [0] * len(javafilelist)
    db = Database()
    #matchobj = re.compile(r'([Ff]ix(ed)?|[Bb]ug) #?[0-9]+')
    matchobj = re.compile(r'([Ff]ix(ed)?|[Bb]ug) ')
    #matchobj = re.compile(r'[Ff]ix(ed)? #?[0-9]+')
    version = ["version LIKE '{}'".format(v) for v in version]
    record = db.SELECT('commits',['commitMessage','fileFilename'],version)
    print(len(record))
    for r in record:
        if not r[1].find('.java'):
            continue
        if matchobj.match(r[0]):
            print(r[0])
            for i,file in enumerate(javafilelist):
                if r[1] == file:
                    label[i] = 1

    with open(wfile,'w') as f:
        for i,_ in enumerate(javafilelist):
            f.write(javafilelist[i]+","+str(label[i])+"\n")
"""

if __name__ == '__main__':
    proj = "Lucene"
    for ver in dsconf[proj]["versions"]:
        projectName = dsconf[proj][ver]
        metricsname = "./dataset/metrics/"+proj+"-"+ver.replace("_",".")+".csv"
        makelabeldata("./dataset/githubsrc/"+projectName,metricsname,config.dataconf['path_labels']+"/"+projectName+".csv")

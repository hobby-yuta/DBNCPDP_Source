from makeohv import makeohv,getOhvDirName
from datasetConfig import dsconf

def makeCPDPohv():
    for trainPRJ in dsconf["projects"]:
        for testPRJ in dsconf["projects"]:
            if trainPRJ != testPRJ:
                makeohv(trainPRJ,dsconf[trainPRJ]["versions"][-1],testPRJ,dsconf[testPRJ]["versions"][-1],trainPRJ,dsconf[trainPRJ]["versions"][-2])
                makeohv(trainPRJ,dsconf[trainPRJ]["versions"][-1],testPRJ,dsconf[testPRJ]["versions"][-1],testPRJ,dsconf[testPRJ]["versions"][-2])

def CPDPdict():
    dict = {}
    for trainPRJ in dsconf["projects"]:
        for testPRJ in dsconf["projects"]:
            if trainPRJ != testPRJ:
                valPRJs = [trainPRJ,testPRJ]
                valVers = [dsconf[trainPRJ]["versions"][-2],dsconf[testPRJ]["versions"][-2]]
                for valPRJ,valVer in zip(valPRJs,valVers):
                    proj = trainPRJ[:2]+"To"+testPRJ[:2]+"Val"+valPRJ[:2]
                    dict[proj] = {}
                    dict[proj]["ohv"] = getOhvDirName(trainPRJ,dsconf[trainPRJ]["versions"][-1],testPRJ,dsconf[testPRJ]["versions"][-1],valPRJ,valVer)
                    dict[proj]["train"] = dsconf[trainPRJ][dsconf[trainPRJ]["versions"][-1]]
                    dict[proj]["test"] = dsconf[testPRJ][dsconf[testPRJ]["versions"][-1]]
                    dict[proj]["val"] = dsconf[valPRJ][valVer]
    return dict


if __name__ == "__main__":
    makeCPDPohv()

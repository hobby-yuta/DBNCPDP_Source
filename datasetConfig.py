
dsconf = {
    "path":"C:/Users/kome/Documents/ibadai/Takahashilab/DBN_Source/dataset/",
    "srcdir":"githubsrc",
    "ohvdir":"githubohv",
    "metricsdir":"metrics",
    "labeldir":"labels",
    "projects":["Xerces","Xalan","Synapse","poi","Lucene"],
    "Xerces": {
        "versions":["1_2","1_3","1_4"],
        "1_2":"Xerces-J_1_2_0",
        "1_3":"Xerces-J_1_3_0",
        "1_4":"Xerces-J_1_4_0"
    },

    "Xalan": {
        "versions":["2_4","2_5","2_6"],
        "2_4":"xalan_2_4_0",
        "2_5":"xalan_2_5_0",
        "2_6":"xalan_2_6_0"
    },

    "Synapse": {
        "versions":["1_0","1_1","1_2"],
        "1_0":"synapse-1_0",
        "1_1":"synapse-1_1",
        "1_2":"synapse-1_2"
    },

    "poi": {
        "versions":["1_5","2_5","3_0"],
        "1_5":"poi-1_5",
        "2_5":"poi-2_5",
        "3_0":"poi-3_0"
    },

    "Lucene": {
        "versions":["2_0","2_2","2_4"],
        "2_0":"lucene-2_0",
        "2_2":"lucene-2_2",
        "2_4":"lucene-2_4"
    },

    "WPDP": {
        "Xerces":{
            "ohv":"Xe1_2Xe1_3Xe1_4",
            "train":"1_3",
            "test":"1_4",
            "val":"1_2",
        },
        "Xalan":{
            "ohv":"Xa2_4Xa2_5Xa2_6",
            "train":"2_5",
            "test":"2_6",
            "val":"2_4"
        },
        "Synapse":{
            "ohv":"Sy1_0Sy1_1Sy1_2",
            "train":"1_1",
            "test":"1_2",
            "val":"1_0"
        },
        "poi":{
            "ohv":"po1_5po2_5po3_0",
            "train":"2_5",
            "test":"3_0",
            "val":"1_5"
        },
        "Lucene":{
            "ohv":"lu2_0lu2_2lu2_4",
            "train":"2_2",
            "test":"2_4",
            "val":"2_0"
        }
    },
    "CPDP": {
        "XeToXaValXe":{
            "ohv":"Xe1_3Xe1_4Xa2_6",
            "train":["Xerces","1_4"],
            "test":["Xalan","2_6"],
            "val":["Xerces","1_3"]
        },
        "XeToXaValXa":{
            "ohv":"Xe1_4Xa2_5Xa2_6",
            "train":["Xerces","1_4"],
            "test":["Xalan","2_6"],
            "val":["Xalan","2_5"]
        },
        "XaToXeValXa":{
            "ohv":"Xe1_4Xa2_5Xa2_6",
            "train":["Xalan","2_6"],
            "test":["Xerces","1_4"],
            "val":["Xalan","2_5"]
        },
        "XaToXeValXe":{
            "ohv":"Xe1_3Xe1_4Xa2_6",
            "train":["Xalan","2_6"],
            "test":["Xerces","1_4"],
            "val":["Xerces","1_3"]
        },
        "XeToSyValXe":{
            "ohv":"Xe1_3Xe1_4Sy1_2",
            "train":["Xerces","1_4"],
            "test":["Synapse","1_2"],
            "val":["Xerces","1_3"]
        },
        "XeToSyValSy":{
            "ohv":"Xe1_4Sy1_1Sy1_2",
            "train":["Xerces","1_4"],
            "test":["Synapse","1_2"],
            "val":["Synapse","1_1"]
        },

    }

}

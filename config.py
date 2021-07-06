import datetime

dbconf = {
    'user':'root',
    'passwd':'ensyu',
    'db':'GitHub',
    'host':'localhost',

    'tables':{
        'commits':{
            'columns':{
                'organization':'varchar(20)',
                'repository':'varchar(20)',
                'version':'varchar(30)',
                'comments_url':'varchar(100)',
                'commitMessage':'varchar(1000)',
                'commitUrl':'varchar(100)',
                'fileFilename':'varchar(100)',
                'fileAddition':'int',
                'fileDeletion':'int',
                'fileChange':'int',
                'fileBlob_url':'varchar(100)',
                'filePatch':'varchar(1000)',
                'filePrevious_filename':'varchar(30)',
                'fileRaw_url':'varchar(100)',
                'fileSha':'varchar(100)',
                'html_url':'varchar(100)',
                'parentsHtml_url':'varchar(100)',
                'sha':'varchar(100)',
                'committerHtml_url':'varchar(100)',
                'authorHtml_url':'varchar(100)',

            }
        }
    }

}

gitconf = {
    'keys':{
        'token':'ec25de45bd6d98893c3bf564254f6ad7ee66f509'
    },
    'log4j_v2':{
        'organization':'apache',
        'repository':'log4j'
    },

    'ant':{
        'organization':'apache',
        'repository':'ant'
    },

    'camel':{
        'organization':'apache',
        'repository':'camel'
    },

    'xerces':{
        'organization':'apache',
        'repository':'xerces2-j'
    }
}

#DBNへ入力するデータの作成オプション
dataconf = {
    #抽出するトークンの最低出現回数
    'threshold':1,
    #シーケンス長
    'max_len':150,
    #シーケンス長を超過したシーケンスを分割する際のマージン
    'split_margin':0,
    #onehotベクトルのデータdict.csv,sequences.csvの入るディレクトリの階層
    'path_ohv':'./dataset/githubohv',
    #ラベルデータが入っているディレクトリ
    'path_labels':'./dataset/labels',

    'path_data_pickle':'./dataset/pickle_data'

}

DNNconf = {
    "threshold":[1,2,3],
    "HiddenLayer":[1,2,3,4,5],
    "Dimension":["40","100"],
    "LearningRate":[0.08,0.15],
    "CLF":["LogisticRegression","NaiveBayes","RandomForest"],
}

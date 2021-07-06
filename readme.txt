DBNの使い方


～～ソースコードとラベル作成用のコミットの入手～～

1.config.pyのgitconf以下に学習したいプロジェクトの組織名とリポジトリ名を追加する
	例:apache/xerces2-j


2.githubの学習したいリポジトリのページを開いて、訓練データのバージョンタグをdatabase.pyのsince_tag,until_tagに写す
	例:Xerces_J_3_0,Xerces_J_2_0
  また、そのバージョンのプロジェクトファイルをダウンロードしておく、できれば"dataset/githubsrc"においておく

3.XamppのMySQLを起動する

4.python database.pyを実行し、コミットデータを取得する。プロジェクトによっては時間がかなりかかることもある。



～～トークンシーケンスデータの作成～～

1.ASTDatasetを使って"訓練したいプロジェクトのパス/src"と"テストデータのプロジェクトのパス/src"で解析する。
	例 ".../DBN_Source/dataset/githubsrc/Xerces-J_1_2_0",".../DBN_Source/dataset/Xerces-J_1_3_0"
	
	dict.csvとsequences.csvが出力される

2.config.pyで指定したpath_ohvの下に1で作成したファイルを入れるためのディレクトリを作る
	例 ".../DBN_source/dataset/githubohv/Xerces-J_1_2To1_3"
	    ￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣
		↑config.dataconf["path_ohv"]
	
	作ったらdict.csvとsequences.csvを入れる


～～ラベルデータの作成～～
1.XamppのMySQLを起動する

2.collectJavaFile.pyのmakeJavaFileDataset関数を実行する。
  引数は ラベル作成元のプロジェクトディレクトリ、ラベルファイルの出力先と名前、プロジェクトのバージョン
  ラベルファイルを出力するディレクトリはconfig.pyのdataconf以下のpath_labelsと同じにする
  このときラベルファイルの名前"***.csv" のファイル名***は "githubsrc/" 以下のディレクトリ名と同じにする


～～DBNで学習する～～
1.DBN.pyのmain内のohvdir,trainPRJ,testPRJをそれぞれ以下のようにする
	ohvdir   = dict.csv,sequence.csvの入っているディレクトリ名
	trainPRJ = 訓練データのプロジェクト名
	testPRJ  = テストデータのプロジェクト名

データがそろっていればpython DBN.pyで学習できる


各パラメータ

エポック数(学習の繰り返し数) epochs
　　100が一般的
　　基本的に多いほうがいい。ただし学習率次第では過学習を起こす
　　動作確認なら5とかでいい

学習率 lr
　　0.01～0.2ぐらい?
　　小さすぎると何も学習できない、大きすぎると過学習を起こす。エポック数に合わせて変えるのが良い 

ドロップアウト率 dropout
　　誤差関数の変化を監視して変化がこれを下回ったら学習を途中で終了する。
　　とりあえず0以外で設定しておくとよい

隠れ層の構成 layer_size
　　隠れ層の数とノード数を決める
　　理論上は隠れ層の数を増やすと表現の幅が広まり、隠れ層のノード数を増やすとより多くの特徴を拾える

活性化関数 activation
　　隠れ層や出力層は入力と重みの積をとってバイアスをかけた後に活性化関数に通して出力をする.
　　今使えるのはsigmoidとRelu1
　　sigmoid 出力層の活性化関数で出力を二値分類の確率にするために使う。
　　Relu1   隠れ層の活性化関数でランプ関数がとる値を0～1に限定した関数、

他にも学習率の変化率などがある
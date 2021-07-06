import pygame.mixer

mp3_path = "./util/rsc/Alarm.mp3"


#音再生処理
def Sound(count=1):
    pygame.mixer.init() #初期化
    pygame.mixer.music.load(mp3_path) #読み込み
    pygame.mixer.music.play(count) #ループ再生（引数を1にすると1回のみ再生
    input()
    pygame.mixer.music.stop() #終了

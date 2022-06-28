# 顔・ランドマーク検出による眠気検出・注意喚起システム

![kaisya_chourei_suit_sleep](https://user-images.githubusercontent.com/67746990/176153000-9046aa09-d31b-4d79-ac35-991f860ae116.png)

# システム概要
- 名称：眠気検出・注意喚起システム
- 目的：運転ドライバーやオンライン授業・会議の出席者の眠気を検出し、視覚的に注意喚起を行うことで居眠り防止を行う。
- システム仕様：顔検出及び、目や鼻等の顔パーツを位置座標で検出する顔ランドマーク検出を適用し、被験者の目の面積を求める。そして、目の面積が閾値より小さければ眠気判定及び注意喚起を行う。

# アルゴリズム詳細
- カメラで取得したリアルタイム映像を読み込む。フレームごとに以下処理を行う。
- 顔検出により顔領域を検出する。検出できればフレームに青色四角形で表示し以下処理を行う。
- 顔領域内で顔ランドマーク検出(目や鼻等を位置座標にて検出する)を行う。
- 目の位置座標から目の面積(ear指標：eyes aspect ratio)を算出し、リスト型キューで保持する。
- キューに保持されている現フレームと前数フレーム分の目の面積(ear指標)で平均値を算出する。
- 上記平均値が事前に設定した閾値より下回っていた場合、眠気判定し注意喚起を行う。
- 以上をフレームに書き出し視覚的に表示する。
# ear指標
$ear =\displaystyle\frac{( | p2 - p6 | + | p3 - p5 | )}{2\times| p1 - p4 |}$

![image](https://user-images.githubusercontent.com/67746990/176149134-a237db68-bb78-4c37-9a23-1423324234c9.png)

# 実行環境について：
- OS：Windows10
- 言語：python
- ライブラリ：opencv,dlib etc.
- ノートPC（ROG Zephyrus G14 GA401IV）
- 外付けカメラ（MAXHUB UC W10）

# 実行結果
目を一定時間閉じると警告文が表示され、注意喚起が行われる。

![output_result](https://user-images.githubusercontent.com/67746990/176146843-ffaddb8f-0ca4-4a5e-aa5e-e450accc061b.gif)

（※被験者は自分。動画編集によりモザイク適用済み）

# 本システム独自性
- 注意喚起表示では人に刺激を与える赤色を使用。赤色はアドレナリンの分泌を活性化させる効果や自律神経を刺激し緊張状態にさせる等の効果があり注意喚起として有効であると考え導入した。また、画面全体を赤色枠で表示すること、赤色で画面中央へ注意文字を表示することで、注意喚起を視覚的に分かりやすく表現した。

- 後述の前数フレーム分の目の面積情報を保持する際には、リスト型のキューを使用した。合計算出や保持数算出などの平均化処理を容易化するためである。

# 本システム問題点
- (ⅰ)瞬きを行うと極端に目の面積が小さくなる。そのため眠気誤検出を誘発してしまう問題がある。

- (ⅱ)被験者の顔が正面から上下左右に少し傾く時には、顔検出が行われず、後続処理ができない問題がある。

- (ⅲ)閾値は定性的な決め方をしているため汎用的ではないこと。

# 問題点に対する改善策
- 問題点(ⅰ)：目の面積を求める際には現フレームと前数フレーム分の目の面積の平均とし、頑健性を向上させることで対処した。(※本システム独自性)

- 問題点(ⅱ)：一般的な問題であり、顔検出モデル精度により左右される。そのため、カメラ台数を増やす(顔正面の上下前後左右に配置し、一番顔が正面に近づくカメラを選択する等)ハードウェア的方法や常に正面を向くように注意を促す心理学的方法等、ソフトウェアのみにこだわらない方法で解決策を検討中である。

- 問題点(ⅲ)：通常時や眠気時の目の面積平均を予め算出し、この情報を基に定量的に閾値を決定する方法を検討中である。

# 参照
- https://qiita.com/mogamin/items/a65e2eaa4b27aa0a1c23
- http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf
- https://qiita.com/mamon/items/bb2334eef596f8cacd9b
- https://qiita.com/RanWensheng/items/d8768395166d041a753a
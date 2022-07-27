"""
predict.py

学習済みのニューラルネットワークを読み込み、画像分類を実施する。
"""

__author__ = 'Daiki Hashimoto (Mizuho Infomation & Research Institute, Inc.)'
__version__ = '1.0.0'
__data__ = 'Feb 24 2020'

import sys
import argparse
from keras.models import load_model
import numpy as np
from PIL import Image  # conda install pillow

# Cifar10のクラス分類のラベル名のリスト
LABELS = ['cat', 'dog']


def parse_argv():
    """
    実行引数のチェック&読み込みを実施
    """
    # make argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', help='input image file path.')
    parser.add_argument("--model-path", default='/opt/work/engine/dataset/cnn_model_weights.hdf5',
                        help='trained model path (default=%(default)s).')

    # parse
    args = parser.parse_args(sys.argv[1:])

    return args


def main():
    """
    メイン関数

    学習済みのニューラルネットワークモデルと画像を読み込み、ニューラルネットワークによる画像の分類結果を返す

    :return:
    """

    # 実行引数のチェック&読み込み
    args = parse_argv()

    # 学習済みモデルの読み込み
    model = load_model(args.model_path)

    # 学習済みモデルの入力サイズを取得
    input_image_shape = (model.input_shape[1], model.input_shape[2])  # (32, 32)

    # 画像の読み込み
    img = Image.open(args.image_path).convert('RGB')

    # 前処理
    # 1. resize: 学習済みモデルの入力サイズに合わせる
    img = img.resize(input_image_shape)
    # 2. PIL.Image -> numpy配列
    img_array = np.array(img)
    # 3. データが [0, 1] の範囲に収まるように正規化
    img_array = img_array.astype('float32') / 255.0
    # 4. reshape: shape=(32, 32, 3) -> shape=(データ数(=1), 32, 32, 3)
    input_array = img_array.reshape((1, ) + input_image_shape + (3, ))

    #  学習済みモデルによる予測
    pred = model.predict(input_array)  # pred shape=(データ数(=1), クラス数(=10))

    # クラスごとのスコア
    for i, label in enumerate(LABELS):
        print('score of {label:10}: {score:0.3f}'.format(label=label, score=pred[0, i]))

    # 予測結果(スコアの最も大きなクラス)
    print('pred: ', LABELS[np.argmax(pred)])


if __name__ == '__main__':
    main()


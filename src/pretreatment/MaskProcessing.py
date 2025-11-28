import matplotlib.pyplot as plt
import cv2
import numpy as np
from config import move_figure, BLACK_LEVEL
from image_util import to_8bit_gamma

# ==========================
# クラス: マスク処理とGUI操作
# ==========================
class MaskProcessor:
    """
    画像の指定領域をマスク処理し、カラーチェッカー領域も切り出して保存するクラス
    GUIで4点をクリックして領域を指定する
    """
    def __init__(self, image_path, output_path, checker_path):
        self.image_path = image_path        # 入力画像パス
        self.output_path = output_path      # マスク画像保存先
        self.checker_path = checker_path    # カラーチェッカー領域保存先
        self.coords = []                    # クリック座標を格納
        self.action = {"next": False, "quit": False}  # 次へ/終了フラグ

        # 画像読み込みと表示用変換
        self.img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        self.img = np.clip(self.img - BLACK_LEVEL, 0, None).astype(np.uint16)
        self.img_display = to_8bit_gamma(self.img)
        self.img_rgb_display = cv2.cvtColor(self.img_display, cv2.COLOR_BGR2RGB)

    # --------------------------
    # マスク処理とカラーチェッカー切り出し
    # --------------------------
    def mask_region_and_save(self):
        """クリックした領域を黒く塗りつぶし、マスク画像とカラーチェッカー領域を保存"""
        mask = np.ones_like(self.img, dtype=np.float32)
        pts = np.array([self.coords], dtype=np.int32)

        # 選択領域を黒く塗りつぶす
        cv2.fillPoly(mask, pts, (0.0, 0.0, 0.0))
        masked_img = (self.img * mask).astype(np.uint16)
        cv2.imwrite(self.output_path, masked_img)
        print(f"Saved masked image to: {self.output_path}")

        # カラーチェッカー領域を切り出して保存
        x, y, w, h = cv2.boundingRect(pts)
        cropped_checker = self.img[y:y+h, x:x+w].astype(np.uint16)
        cv2.imwrite(self.checker_path, cropped_checker)
        print(f"Saved cropped checker region to: {self.checker_path}")

        # GUI上でもマスク結果を表示
        masked_display = to_8bit_gamma(cv2.imread(self.output_path, cv2.IMREAD_UNCHANGED).astype(np.float32))
        self.ax.imshow(cv2.cvtColor(masked_display, cv2.COLOR_BGR2RGB))
        self.ax.set_title("Press Enter to continue or q to quit")
        self.fig.canvas.draw()

    # --------------------------
    # クリックイベント
    # --------------------------
    def onclick(self, event):
        """クリックした座標を保存し、4点揃ったらマスク処理を実行"""
        if event.xdata and event.ydata and len(self.coords) < 4:
            x, y = int(event.xdata), int(event.ydata)
            self.coords.append((x, y))
            self.ax.plot(x, y, 'ro')
            self.ax.text(x+5, y-5, f"{len(self.coords)}", color="red", fontsize=12)
            self.fig.canvas.draw()

            if len(self.coords) == 4:
                self.mask_region_and_save()

    # --------------------------
    # キーイベント
    # --------------------------
    def onkey(self, event):
        """rでリセット、スペースで次へ、qで終了"""
        if event.key == 'r':
            self.coords.clear()
            self.ax.clear()
            self.ax.imshow(self.img_rgb_display)
            self.ax.set_title("Click 4 points (Press 'r' to reset)")
            self.fig.canvas.draw()
        elif event.key == ' ':
            self.action["next"] = True
            plt.close()
        elif event.key == 'q':
            self.action["quit"] = True
            plt.close()

    # --------------------------
    # GUI実行
    # --------------------------
    def run(self):
        """GUIを表示してユーザーがクリックするのを待つ"""
        self.fig, self.ax = plt.subplots(figsize=(10, 9))
        move_figure(self.fig, 0, 20)
        self.ax.imshow(self.img_rgb_display)
        self.ax.set_title("Click 4 points (Press 'r' to reset)")

        # イベント接続
        self.fig.canvas.mpl_connect("button_press_event", self.onclick)
        self.fig.canvas.mpl_connect("key_press_event", self.onkey)

        plt.show()
        return self.action

# ==========================
# ラッパー関数
# ==========================
def MaskProcessing(image_path, output_path, checker_path):
    """
    MaskProcessorクラスを呼び出すラッパー関数
    :return: {"next": True/False, "quit": True/False}
    """
    processor = MaskProcessor(image_path, output_path, checker_path)
    return processor.run()

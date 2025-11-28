import subprocess
import json

def extract_dng_metadata_exiftool(dng_path):
    result = subprocess.run(
        ["exiftool", "-j", dng_path],
        capture_output=True,
        text=True,
        encoding="utf-8"   # ここを追加
    )

    if result.returncode != 0:
        print("❌ exiftool 実行エラー:", result.stderr)
        return None

    try:
        metadata = json.loads(result.stdout)
        return metadata[0] if metadata else None
    except json.JSONDecodeError:
        print("❌ JSONデコードエラー。出力内容:", result.stdout[:500])
        return None

if __name__ == "__main__":
    dng_path = 'C:/Users/taiki/Desktop/test/IMG_4017.dng'
    meta = extract_dng_metadata_exiftool(dng_path)

    # JSONファイルに保存
    with open("metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("✅ metadata.json に保存しました")

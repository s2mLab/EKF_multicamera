from pathlib import Path

from PIL import Image

input_dir = Path("/Users/mickaelbegon/Documents/Playground/inputs/images/1_partie_0429")

QUALITY = 85  # ajuste entre 75–90 selon ton besoin
DELETE_SOURCE = True  # True pour supprimer l'image source après conversion réussie

valid_ext = {".jpg", ".jpeg", ".png"}

for img_path in input_dir.rglob("*"):
    if not img_path.is_file() or img_path.suffix.lower() not in valid_ext:
        continue

    try:
        with Image.open(img_path) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            elif img.mode != "RGB":
                img = img.convert("RGB")

            # Le JPEG est sauvegardé dans le même dossier que l'image source
            out_path = img_path.with_suffix(".jpg")

            # Si le fichier de sortie est le même que la source (.jpg déjà existant),
            # on passe par un fichier temporaire pour éviter les conflits.
            if out_path.resolve() == img_path.resolve():
                temp_out_path = img_path.with_name(img_path.stem + "_temp_compressed.jpg")
                img.save(temp_out_path, "JPEG", quality=QUALITY, optimize=True, progressive=True)
                size_before = img_path.stat().st_size / 1e6
                temp_out_path.replace(out_path)
                size_after = out_path.stat().st_size / 1e6

                print(f"✔ {img_path.name}: {size_before:.2f} MB → {size_after:.2f} MB")

            else:
                img.save(out_path, "JPEG", quality=QUALITY, optimize=True, progressive=True)

                size_before = img_path.stat().st_size / 1e6
                size_after = out_path.stat().st_size / 1e6

                if DELETE_SOURCE:
                    img_path.unlink()
                    print(f"✔ {img_path.name}: {size_before:.2f} MB → {size_after:.2f} MB (source supprimée)")
                else:
                    print(f"✔ {img_path.name}: {size_before:.2f} MB → {size_after:.2f} MB")

    except Exception as e:
        print(f"✖ {img_path.name}: {e}")

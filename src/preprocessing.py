
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATASET_PATH = Path("/home/mouad/BrainScan_deep_learning_cnn/data/row") 
IMG_SIZE = 224
TEST_SIZE = 0.20
RANDOM_STATE = 42

VALID_EXTENSIONS = {".jpeg", ".jpg", ".bmp", ".png"}

np.random.seed(RANDOM_STATE)

if not DATASET_PATH.is_dir():
    raise FileNotFoundError(
        f"Dataset introuvable : {DATASET_PATH.resolve()}"
    )


# ============================================================
# 2. EXPLORATION DES CLASSES
# ============================================================

class_dirs = sorted(
    folder for folder in DATASET_PATH.iterdir()
    if folder.is_dir()
)

if len(class_dirs) < 2:
    raise ValueError(
        "Le dataset doit contenir au moins deux classes."
    )

class_names = [folder.name for folder in class_dirs]

print("=" * 60)
print("CLASSES DU DATASET")
print("=" * 60)

for name in class_names:
    print("-", name)


# ============================================================
# 3. CHARGEMENT ET VALIDATION DES IMAGES
# ============================================================

images = []
labels = []

errors = []
ignored_files = []

for class_dir in class_dirs:

    class_name = class_dir.name

    for image_path in sorted(class_dir.rglob("*")):

        if not image_path.is_file():
            continue

        # Vérifier les extensions
        if image_path.suffix.lower() not in VALID_EXTENSIONS:
            ignored_files.append(str(image_path))
            continue

        try:
            # Lire l'image sans supprimer ses canaux
            image = cv2.imread(
                str(image_path),
                cv2.IMREAD_UNCHANGED
            )

            if image is None:
                raise ValueError("Image illisible ou corrompue")

            # Images grayscale : 1 canal vers RGB
            if image.ndim == 2:
                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_GRAY2RGB
                )

            # Images couleur : BGR vers RGB
            elif image.ndim == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                )

            # Images avec canal alpha : BGRA vers RGB
            elif image.ndim == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGRA2RGB
                )

            else:
                raise ValueError(
                    f"Format non pris en charge : {image.shape}"
                )

            # Redimensionner les images
            image = cv2.resize(
                image,
                (IMG_SIZE, IMG_SIZE),
                interpolation=cv2.INTER_AREA
            )

            # Ajouter l'image et son label au même indice
            images.append(image)
            labels.append(class_name)

        except Exception as error:
            errors.append((str(image_path), str(error)))


# Vérifier que le dataset n'est pas vide
if len(images) == 0:
    raise ValueError("Aucune image valide trouvée.")

# Conversion en tableaux NumPy
X = np.stack(images).astype(np.uint8)
y = np.array(labels)

assert len(X) == len(y)

print("\n" + "=" * 60)
print("RÉSULTAT DU CHARGEMENT")
print("=" * 60)
print("Nombre d'images :", len(X))
print("Nombre de labels :", len(y))
print("Shape de X :", X.shape)
print("Shape de y :", y.shape)
print("Fichiers ignorés :", len(ignored_files))
print("Images en erreur :", len(errors))


# ============================================================
# 4. NOMBRE D'IMAGES PAR CLASSE
# ============================================================

class_counts = Counter(y)

print("\nDistribution des classes :")

for class_name in class_names:
    print(f"{class_name}: {class_counts[class_name]} images")

plt.figure(figsize=(10, 5))

plt.bar(
    list(class_counts.keys()),
    list(class_counts.values())
)

plt.title("Nombre d'images par classe")
plt.xlabel("Classes")
plt.ylabel("Nombre d'images")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# ============================================================
# 6. VÉRIFIER L'ÉQUILIBRE DES CLASSES
# ============================================================

counts = np.array([
    class_counts[name] for name in class_names
])

min_count = counts.min()
max_count = counts.max()

imbalance_ratio = max_count / min_count

print("\n" + "=" * 60)
print("ANALYSE DE L'ÉQUILIBRE")
print("=" * 60)

print("Classe la moins représentée :", min_count)
print("Classe la plus représentée :", max_count)
print(f"Ratio max/min : {imbalance_ratio:.2f}")

if imbalance_ratio >= 1.5:
    print("Déséquilibre potentiel détecté.")
    print("Une pondération des classes sera évaluée.")
else:
    print("Les classes sont relativement équilibrées.")


# ============================================================
# 7. ENCODAGE DES LABELS
# ============================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nCorrespondance des labels :")

for index, class_name in enumerate(label_encoder.classes_):
    print(f"{class_name} -> {index}")

print("Exemple de labels encodés :", y_encoded[:10])


# ============================================================
# 8. DIVISION TRAIN / TEST
# ============================================================

# Vérifier que chaque classe possède suffisamment d'images
encoded_counts = np.bincount(y_encoded)

if encoded_counts.min() < 2:
    raise ValueError(
        "Chaque classe doit contenir au moins deux images "
        "pour une division stratifiée. Collectez plus d'images."
    )

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_encoded
)

print("\n" + "=" * 60)
print("DIVISION TRAIN / TEST")
print("=" * 60)

print("X_train :", X_train.shape)
print("X_test  :", X_test.shape)
print("y_train :", y_train.shape)
print("y_test  :", y_test.shape)


# ============================================================
# 9. NORMALISATION DES PIXELS : [0, 1]
# ============================================================

X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

print("\nNormalisation terminée.")

print("Minimum X_train :", X_train.min())
print("Maximum X_train :", X_train.max())

print("Minimum X_test :", X_test.min())
print("Maximum X_test :", X_test.max())




# ============================================================
# 10. RÉSULTATS FINAUX
# ============================================================

print("\n" + "=" * 60)
print("PRÉTRAITEMENT TERMINÉ")
print("=" * 60)

print("Nombre de classes :", len(class_names))
print("Classes :", list(label_encoder.classes_))
print("Train :", len(X_train), "images")
print("Test  :", len(X_test), "images")
print("Dimensions :", IMG_SIZE, "x", IMG_SIZE, "x 3")

print("\nVariables disponibles pour le CNN :")
print("X_train, y_train")
print("X_test, y_test")
print("label_encoder")
print("class_weights")

# Sauvegarder les données prétraitées
OUTPUT_PATH = Path("/data/processed_data")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    OUTPUT_PATH / "medical_dataset.npz",
    X_train=X_train,    
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    classes=label_encoder.classes_
)

print(
    "\nDonnées sauvegardées dans :",
    (OUTPUT_PATH / "medical_dataset.npz").resolve()
)

# Sauvegarder les erreurs pour inspection
if errors:
    with open(
        OUTPUT_PATH / "image_errors.txt",
        "w",
        encoding="utf-8"
    ) as file:
        for image_path, message in errors:
            file.write(f"{image_path} | {message}\n")

print("Fin du script.")

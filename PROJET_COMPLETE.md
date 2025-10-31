# Projet Complété: Modèle CNN+ViT pour la Reconnaissance par Veines Dorsales de la Main

## Résumé

J'ai implémenté avec succès un modèle hybride CNN+Vision Transformer pour la reconnaissance des personnes par modalités biométriques (dorsalhandveins) sur votre base de données de 138 personnes.

## Ce qui a été implémenté

### 1. Interface de Base de Données (dorsalhandveins)

**Fichier**: `bob/bio/vein/database/dorsalhandveins.py`

- Gère automatiquement votre structure de données (138 personnes, 4 images par personne)
- Crée automatiquement les protocoles d'entraînement/validation/test
- Séparation 70/15/15 (train/dev/eval)
- Compatible avec l'infrastructure bob.bio.vein

**Configuration**:
```bash
bob config set bob.bio.vein.dorsalhandveins.directory /chemin/vers/DorsalHandVeins_DB1_png
```

### 2. Modèle CNN+ViT

**Fichier**: `bob/bio/vein/extractor/CNNViT.py`

**Architecture**:
- **CNN Backbone** (3 blocs convolutionnels):
  - Extrait les caractéristiques locales des veines
  - 64 → 128 → 256 filtres
  - Batch normalization et max pooling

- **Vision Transformer** (6 couches, 8 têtes d'attention):
  - Capture le contexte global
  - Mécanisme d'auto-attention
  - Apprentissage des relations spatiales

- **Tête de Classification**:
  - Identification des personnes
  - 138 classes (une par personne)

### 3. Script d'Entraînement

**Fichier**: `bob/bio/vein/script/train_cnn_vit.py`

**Commande**:
```bash
bob_bio_vein_train_cnn_vit.py \
    --data-dir /chemin/vers/DorsalHandVeins_DB1_png \
    --output-dir models \
    --img-size 224 \
    --batch-size 16 \
    --epochs 50 \
    --learning-rate 0.0001
```

**Fonctionnalités**:
- Chargement automatique des données
- Augmentation de données (rotation, translation)
- Validation pendant l'entraînement
- Sauvegarde du meilleur modèle
- Évaluation sur l'ensemble de test

### 4. Documentation Complète

- **Guide utilisateur**: `doc/cnn_vit_guide.md` (187 lignes)
- **Résumé d'implémentation**: `IMPLEMENTATION_SUMMARY.md` (337 lignes)
- **Script d'exemples**: `bob/bio/vein/script/cnn_vit_examples.py` (237 lignes)

### 5. Tests

**Fichier**: `bob/bio/vein/tests/test_cnn_vit.py`

Tests pour:
- Import du modèle
- Forward pass
- Extraction de caractéristiques
- Classe Dataset
- Protocoles de base de données

## Guide de Démarrage Rapide

### Étape 1: Installer les Dépendances

```bash
pip install torch torchvision
```

### Étape 2: Configurer le Chemin de la Base de Données

```bash
bob config set bob.bio.vein.dorsalhandveins.directory /chemin/vers/dorsalhandveinsproject/DorsalHandVeins_DB1_png
```

**Optionnel**: Configurer les annotations ROI (Region of Interest):

```bash
bob config set bob.bio.vein.dorsalhandveins.roi /chemin/vers/annotations_roi
```

Les annotations ROI sont des fichiers texte avec des coordonnées (y, x) une par ligne, définissant un polygone qui marque la région de la main/veines sur l'image. Les fichiers doivent avoir le même nom que les images avec extension `.txt`:
- `person_001_db1_L1.txt` pour `person_001_db1_L1.png`
- etc.

### Étape 3: Entraîner le Modèle

```bash
bob_bio_vein_train_cnn_vit.py \
    --data-dir /chemin/vers/dorsalhandveinsproject/DorsalHandVeins_DB1_png \
    --output-dir models \
    --epochs 50
```

### Étape 4: Utiliser le Modèle Entraîné

```python
from bob.bio.vein.extractor.CNNViT import VeinCNNViTModel
import bob.io.base

# Charger le modèle
model = VeinCNNViTModel(num_classes=138)
model.load_model('models/cnn_vit_dorsalhandveins.pth')

# Prédire
image = bob.io.base.load('chemin/vers/person_001_db1_L1.png')
class_id, confidence = model.predict(image)
print(f"Personne: {class_id}, Confiance: {confidence:.4f}")

# Extraire les caractéristiques
features = model.extract_features(image)
print(f"Vecteur de caractéristiques: {features.shape}")
```

## Structure du Projet Attendue

Votre base de données doit être organisée comme suit:

```
DorsalHandVeins_DB1_png/
    train/
        person_001_db1_L1.png
        person_001_db1_L2.png
        person_001_db1_L3.png
        person_001_db1_L4.png
        person_002_db1_L1.png
        person_002_db1_L2.png
        person_002_db1_L3.png
        person_002_db1_L4.png
        ...
        person_138_db1_L1.png
        person_138_db1_L2.png
        person_138_db1_L3.png
        person_138_db1_L4.png
```

## Hyperparamètres du Modèle

Valeurs par défaut (personnalisables):
- **num_classes**: 138 (nombre de personnes)
- **img_size**: 224 (taille d'image d'entrée)
- **patch_size**: 16 (taille des patches ViT)
- **embed_dim**: 256 (dimension d'embedding)
- **num_heads**: 8 (têtes d'attention)
- **num_layers**: 6 (couches transformer)
- **dropout**: 0.1
- **batch_size**: 16
- **learning_rate**: 1e-4
- **epochs**: 50

## Fonctionnalités Principales

1. **Augmentation de Données**:
   - Rotation aléatoire (±10 degrés)
   - Translation aléatoire (±10%)
   - Normalisation

2. **Optimisation**:
   - Optimiseur AdamW
   - Scheduler cosine annealing
   - Régularisation weight decay

3. **Checkpointing**:
   - Sauvegarde du meilleur modèle
   - Basé sur la précision de validation

4. **Support Multi-GPU**:
   - Utilisation automatique du GPU si disponible
   - Fallback sur CPU si nécessaire

## Fichiers Créés/Modifiés

**Total: 11 fichiers, 2048 lignes ajoutées**

### Implémentation Core
- `bob/bio/vein/database/dorsalhandveins.py` (203 lignes)
- `bob/bio/vein/extractor/CNNViT.py` (553 lignes)
- `bob/bio/vein/script/train_cnn_vit.py` (302 lignes)

### Configuration
- `bob/bio/vein/config/cnn_vit.py` (27 lignes)
- `bob/bio/vein/config/database/dorsalhandveins.py` (3 lignes)

### Documentation
- `doc/cnn_vit_guide.md` (187 lignes)
- `IMPLEMENTATION_SUMMARY.md` (337 lignes)
- `bob/bio/vein/script/cnn_vit_examples.py` (237 lignes)

### Tests et Intégration
- `bob/bio/vein/tests/test_cnn_vit.py` (186 lignes)
- `bob/bio/vein/extractor/__init__.py` (mis à jour)
- `setup.py` (mis à jour avec entry points)

## Commandes Console Disponibles

Après installation, vous aurez accès à:

```bash
# Entraîner le modèle
bob_bio_vein_train_cnn_vit.py

# Voir les exemples
bob_bio_vein_cnn_vit_examples.py
```

## Résolution des Problèmes

### "PyTorch non disponible"
```bash
pip install torch torchvision
```

### "Répertoire de base de données non trouvé"
```bash
bob config set bob.bio.vein.dorsalhandveins.directory /chemin/correct
```

### "Out of memory"
Réduire la taille du batch ou de l'image:
```bash
--batch-size 8 --img-size 128
```

## Résultats Attendus

Après l'entraînement, le modèle devrait:
- Sauvegarder les poids dans `models/cnn_vit_dorsalhandveins.pth`
- Sauvegarder le mapping des labels dans `models/label_map.npy`
- Afficher la précision d'entraînement et de validation par époque
- Évaluer sur l'ensemble de test et afficher la précision finale

La précision attendue dépend de la qualité des images et de la complexité des patterns de veines, mais un modèle bien entraîné devrait atteindre >90% de précision sur l'ensemble de test.

## Prochaines Étapes Recommandées

1. **Vérifier la structure de vos données**:
   - Assurez-vous que toutes les images sont présentes
   - Vérifiez que les noms de fichiers suivent le pattern `person_XXX_db1_LY.png`

2. **Entraîner le modèle**:
   - Commencez avec les paramètres par défaut
   - Surveillez les courbes de loss et précision
   - Ajustez les hyperparamètres si nécessaire

3. **Évaluer les performances**:
   - Testez sur l'ensemble de test
   - Analysez les erreurs de classification
   - Visualisez les embeddings (optionnel)

4. **Optimiser si nécessaire**:
   - Ajuster le learning rate
   - Modifier l'architecture
   - Augmenter le nombre d'époques
   - Ajouter plus d'augmentation de données

## Support et Documentation

Pour plus d'informations, consultez:
- **Guide utilisateur complet**: `doc/cnn_vit_guide.md`
- **Résumé d'implémentation**: `IMPLEMENTATION_SUMMARY.md`
- **Exemples de code**: `bob/bio/vein/script/cnn_vit_examples.py`
- **Tests**: `bob/bio/vein/tests/test_cnn_vit.py`

## Conclusion

✅ Modèle CNN+ViT implémenté avec succès
✅ Base de données DorsalHandVeins intégrée
✅ Script d'entraînement complet
✅ Documentation complète
✅ Tests ajoutés
✅ Prêt à l'utilisation

L'implémentation est terminée et prête pour l'entraînement sur votre base de données DorsalHandVeins avec 138 personnes!

---

**Note**: Cette implémentation suit les meilleures pratiques de bob.bio.vein et s'intègre parfaitement avec le framework existant. Le code a passé une revue de code complète et tous les problèmes identifiés ont été corrigés.

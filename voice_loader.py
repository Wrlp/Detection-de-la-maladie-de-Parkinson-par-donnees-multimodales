"""
Loader pour le dataset vocal Parkinson - Utiliser dans vos propres scripts
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional


class VoiceDataset:
    """Classe pour charger et gérer le dataset vocal Parkinson"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialiser le loader
        
        Args:
            data_dir: Répertoire contenant les données. Si None, utilise data/voice/
        """
        if data_dir is None:
            # Déduire le chemin depuis ce fichier
            data_dir = Path(__file__).parent / "data" / "voice"
        
        self.data_dir = Path(data_dir)
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.dataset_type = None
    
    def load(self, use_merged: bool = True, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Charger le dataset
        
        Args:
            use_merged (bool): Si True, utilise le dataset fusionné. Si False, utilise l'original.
            normalize (bool): Si True, normalise les features avec StandardScaler.
        
        Returns:
            Tuple de (X_train, X_test, y_train, y_test)
        
        Exemple:
            >>> loader = VoiceDataset()
            >>> X_train, X_test, y_train, y_test = loader.load(use_merged=True)
        """
        
        # Déterminer les fichiers à charger
        suffix = "_merged" if use_merged else ""
        train_file = self.data_dir / f"train_data{suffix}.txt"
        test_file = self.data_dir / f"test_data{suffix}.txt"
        
        # Vérifier que les fichiers existent
        if not train_file.exists():
            raise FileNotFoundError(f"Fichier train non trouvé: {train_file}")
        if not test_file.exists():
            raise FileNotFoundError(f"Fichier test non trouvé: {test_file}")
        
        # Charger les données
        print(f"Chargement du dataset {'FUSIONNÉ' if use_merged else 'ORIGINAL'}...")
        df_train = pd.read_csv(train_file, header=None)
        df_test = pd.read_csv(test_file, header=None)
        
        # Séparer features et labels
        self.X_train = df_train.iloc[:, :-1].values
        self.y_train = df_train.iloc[:, -1].values.astype(int)
        self.X_test = df_test.iloc[:, :-1].values
        self.y_test = df_test.iloc[:, -1].values.astype(int)
        
        self.dataset_type = "FUSIONNÉ" if use_merged else "ORIGINAL"
        
        # Normaliser si demandé
        if normalize:
            self._normalize()
        
        # Afficher les infos
        self._print_info()
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def _normalize(self):
        """Normaliser les données avec StandardScaler"""
        self.scaler = StandardScaler() # Ajout 
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
    
    def _print_info(self):
        """Afficher les informations du dataset chargé"""
        print(f"   Dataset type: {self.dataset_type}")
        print(f"   ✓ Train: {self.X_train.shape[0]} samples × {self.X_train.shape[1]} features")
        print(f"   ✓ Test: {self.X_test.shape[0]} samples × {self.X_test.shape[1]} features")
        
        n_healthy_train = sum(self.y_train == 0)
        n_sick_train = sum(self.y_train == 1)
        n_healthy_test = sum(self.y_test == 0)
        n_sick_test = sum(self.y_test == 1)
        
        print(f"   ✓ Classes train: {n_healthy_train} sains, {n_sick_train} malades")
        print(f"   ✓ Classes test: {n_healthy_test} sains, {n_sick_test} malades")
    
    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Retourner les données d'entraînement"""
        if self.X_train is None:
            raise ValueError("Vous devez d'abord appeler load()")
        return self.X_train, self.y_train
    
    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Retourner les données de test"""
        if self.X_test is None:
            raise ValueError("Vous devez d'abord appeler load()")
        return self.X_test, self.y_test
    
    def get_all_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Retourner toutes les données"""
        if self.X_train is None:
            raise ValueError("Vous devez d'abord appeler load()")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_class_distribution(self) -> dict:
        """Retourner la distribution des classes"""
        if self.y_train is None:
            raise ValueError("Vous devez d'abord appeler load()")
        
        return {
            'train': {
                'healthy': int(sum(self.y_train == 0)),
                'sick': int(sum(self.y_train == 1))
            },
            'test': {
                'healthy': int(sum(self.y_test == 0)),
                'sick': int(sum(self.y_test == 1))
            }
        }
    
    def info(self) -> str:
        """Retourner les informations du dataset sous forme de string"""
        if self.X_train is None:
            return "Dataset non chargé. Appelez load() d'abord."
        
        dist = self.get_class_distribution()
        info_str = f"""
Dataset: {self.dataset_type}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Train:
  Samples: {self.X_train.shape[0]}
  Features: {self.X_train.shape[1]}
  Sains: {dist['train']['healthy']}
  Malades: {dist['train']['sick']}

Test:
  Samples: {self.X_test.shape[0]}
  Features: {self.X_test.shape[1]}
  Sains: {dist['test']['healthy']}
  Malades: {dist['test']['sick']}
        """
        return info_str


# Fonction helper simplifiée
def load_voice_data(use_merged: bool = True, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fonction simple pour charger rapidement le dataset
    
    Args:
        use_merged: Utiliser le dataset fusionné (True) ou original (False)
        normalize: Normaliser les données (True/False)
    
    Returns:
        Tuple de (X_train, X_test, y_train, y_test)
    
    Exemple:
        >>> X_train, X_test, y_train, y_test = load_voice_data()
    """
    loader = VoiceDataset()
    return loader.load(use_merged=use_merged, normalize=normalize)


# ============================================================================
# EXEMPLES D'UTILISATION
# ============================================================================

if __name__ == "__main__":
    
    print("=" * 70)
    print("EXEMPLES D'UTILISATION DU LOADER VOICE")
    print("=" * 70)
    
    # Exemple 1: Utiliser la fonction simple
    print("\n📝 EXEMPLE 1: Utiliser la fonction simple")
    print("-" * 70)
    X_train, X_test, y_train, y_test = load_voice_data(use_merged=True)
    print(f"Shapes: X_train={X_train.shape}, y_train={y_train.shape}")
    
    # Exemple 2: Utiliser la classe (plus flexible)
    print("\n📝 EXEMPLE 2: Utiliser la classe VoiceDataset")
    print("-" * 70)
    loader = VoiceDataset()
    X_train, X_test, y_train, y_test = loader.load(use_merged=True)
    print(loader.info())
    
    # Exemple 3: Obtenir les informations
    print("\n📝 EXEMPLE 3: Obtenir les informations du dataset")
    print("-" * 70)
    dist = loader.get_class_distribution()
    print(f"Distribution des classes: {dist}")
    
    # Exemple 4: Charger le dataset original
    print("\n📝 EXEMPLE 4: Charger le dataset ORIGINAL")
    print("-" * 70)
    loader2 = VoiceDataset()
    X_train, X_test, y_train, y_test = loader2.load(use_merged=False)
    
    print("\n✅ Tous les exemples fonctionnent!")
